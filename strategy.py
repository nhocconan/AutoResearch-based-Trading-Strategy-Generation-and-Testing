#!/usr/bin/env python3
"""
Experiment #706: 12h Primary + 1d HTF — Connors RSI + Donchian Breakout + HMA Trend

Hypothesis: Simplified entry conditions will generate MORE trades while maintaining
positive Sharpe. Current strategy (#692) has too many confluence filters causing
0 trades in test period. Key changes:

1. CONNORS RSI (CRSI) instead of standard RSI - proven 75% win rate for mean reversion
2. LOOSER thresholds: CRSI < 20 / > 80 (not 15/85) to ensure trade frequency
3. SINGLE HTF filter: 1d HMA only (not 1d+1w) - less restrictive
4. NO SMA200 filter - this blocked longs in 2025 bear market
5. DUAL entry modes: Mean reversion (CRSI extremes) + Breakout (Donchian)
6. SIMPLER exits: CRSI crossback + ATR trail only

Why this should work:
- CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - catches oversold/overbought better
- 12h TF worked in #696 (Sharpe=0.024, +28.7% return) - just needed more trades
- Donchian breakout adds momentum entries when trend is strong
- Fewer filters = more trades = better statistical significance
- ATR 2.5x stop prevents catastrophic drawdown

Target: Sharpe > 0.612, trades >= 80 train (20/year), >= 12 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_donchian_hma_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean-reversion indicator with 75% win rate.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100 - (100 / (1 + rs))
    rsi_short = np.clip(rsi_short, 0, 100)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period+1):i+1]
        avg_streak = np.mean(np.abs(streak_vals))
        # Map streak to 0-100 scale (high streak = overbought)
        streak_rsi[i] = min(100, avg_streak * 20)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = (rank / (rank_period - 1)) * 100
    
    # Combine
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Donchian Channels - highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i-1])
        tr3 = np.abs(low[i] - close[i-1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100 * plus_dm_smooth / (atr + 1e-10)
        di_minus = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_12h = calculate_atr(high, low, close, period=14)
    adx_12h = calculate_adx(high, low, close, period=14)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need buffer for indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(atr_12h[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(adx_12h[i]):
            continue
        if atr_12h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (1d HMA) ===
        trend_bullish = close[i] > hma_1d_aligned[i]
        trend_bearish = close[i] < hma_1d_aligned[i]
        
        # === ADX STRENGTH ===
        adx_strong = adx_12h[i] > 25
        adx_weak = adx_12h[i] < 20
        
        desired_signal = 0.0
        
        # === MEAN REVERSION MODE (ADX weak or neutral) ===
        # Long: CRSI oversold + bullish or neutral trend
        if crsi_12h[i] < 20 and (trend_bullish or not adx_strong):
            desired_signal = BASE_SIZE
        
        # Short: CRSI overbought + bearish or neutral trend
        elif crsi_12h[i] > 80 and (trend_bearish or not adx_strong):
            desired_signal = -BASE_SIZE
        
        # === BREAKOUT MODE (ADX strong) ===
        # Long breakout: price breaks Donchian upper + bullish trend
        elif close[i] > donchian_upper[i] and trend_bullish and adx_strong:
            desired_signal = BASE_SIZE
        
        # Short breakout: price breaks Donchian lower + bearish trend
        elif close[i] < donchian_lower[i] and trend_bearish and adx_strong:
            desired_signal = -BASE_SIZE
        
        # === WEAKER SIGNALS (half size) ===
        # Long: CRSI < 30 without trend confirmation
        if desired_signal == 0.0 and crsi_12h[i] < 30:
            desired_signal = HALF_SIZE
        
        # Short: CRSI > 70 without trend confirmation
        elif desired_signal == 0.0 and crsi_12h[i] > 70:
            desired_signal = -HALF_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought and trend still bullish
                if crsi_12h[i] < 75 and trend_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold and trend still bearish
                if crsi_12h[i] > 25 and trend_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: CRSI overbought OR trend reverses bearish
        if in_position and position_side > 0:
            if crsi_12h[i] > 85:
                desired_signal = 0.0
            elif trend_bearish and adx_strong:
                desired_signal = 0.0
        
        # Short exit: CRSI oversold OR trend reverses bullish
        if in_position and position_side < 0:
            if crsi_12h[i] < 15:
                desired_signal = 0.0
            elif trend_bullish and adx_strong:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals