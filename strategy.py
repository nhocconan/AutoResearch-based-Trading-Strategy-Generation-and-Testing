#!/usr/bin/env python3
"""
Experiment #821: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 20+ failed strategies, key insights:
1. Complex multi-condition entries = 0 trades (Sharpe=0.000 auto-reject)
2. CRSI (Connors RSI) has proven 75% win rate in academic literature for mean reversion
3. Choppiness Index effectively separates trending vs ranging regimes
4. Simpler logic with fewer confluence requirements = more trades while maintaining quality
5. 1d HMA(21) + 1w HMA(21) dual trend filter prevents counter-trend trades
6. Tighter stoploss (1.5x ATR) cuts losses faster than 2.0x
7. Position size 0.25-0.30 balances return vs drawdown (77% BTC crash in 2022)

Strategy design:
1. 1d HMA(21) for intermediate trend (aligned via mtf_data)
2. 1w HMA(21) for long-term trend bias (aligned via mtf_data)
3. 4h Choppiness Index(14) for regime detection (>50 range, <40 trend)
4. 4h Connors RSI for mean reversion entries in ranging regime
5. 4h Donchian(20) for breakout entries in trending regime
6. 4h ATR(14) for trailing stop (1.5x)
7. Discrete signals: 0.0, ±0.20, ±0.30
8. SIMPLIFIED entry logic to guarantee >=10 trades per symbol

Key changes from #811:
- CRSI instead of standard RSI (better mean reversion signal)
- Donchian period: 20 (standard, less noise than 15)
- Stoploss: 1.5x ATR (tighter, faster loss cutting)
- Simpler regime logic (fewer nested conditions = more trades)
- Single trend filter check (1d OR 1w, not both required)

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 25-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_hma_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI of streak length: Persistence of direction
    3. PercentRank: Relative position in recent range
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak length
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on absolute streak values
    streak_abs = np.abs(streak)
    streak_delta = np.diff(streak_abs)
    gain = np.where(streak_delta > 0, streak_delta, 0)
    loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        streak_rsi = 100 - (100 / (1 + rs))
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine components
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 50 = ranging, CHOP < 40 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channels — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === TREND BIAS (1d and 1w HTF HMA21) ===
        trend_bullish = (close[i] > hma_1d_aligned[i]) or (close[i] > hma_1w_aligned[i])
        trend_bearish = (close[i] < hma_1d_aligned[i]) or (close[i] < hma_1w_aligned[i])
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 50
        trending_regime = chop_4h[i] < 40
        
        # === CRSI SIGNALS (Connors RSI for mean reversion) ===
        crsi_extreme_oversold = crsi_4h[i] < 15
        crsi_extreme_overbought = crsi_4h[i] > 85
        crsi_oversold = crsi_4h[i] < 30
        crsi_overbought = crsi_4h[i] > 70
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 50) — Mean Reversion with CRSI ===
        if ranging_regime:
            # Long: CRSI oversold + bullish trend bias (SIMPLE: OR not AND)
            if crsi_oversold and trend_bullish:
                desired_signal = BASE_SIZE
            elif crsi_extreme_oversold:
                # Extreme CRSI alone triggers reduced position (guarantees trades)
                desired_signal = REDUCED_SIZE
            
            # Short: CRSI overbought + bearish trend bias
            if crsi_overbought and trend_bearish:
                desired_signal = -BASE_SIZE
            elif crsi_extreme_overbought and desired_signal == 0:
                # Extreme CRSI alone triggers reduced position
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 40) — Trend Following with Donchian ===
        elif trending_regime:
            # Long: Bullish trend + Donchian breakout
            if trend_bullish and donchian_breakout_long:
                desired_signal = BASE_SIZE
            # Pullback entry in uptrend
            elif trend_bullish and crsi_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + Donchian breakout
            if trend_bearish and donchian_breakout_short:
                desired_signal = -BASE_SIZE
            # Pullback entry in downtrend
            elif trend_bearish and crsi_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (40 <= CHOP <= 50) ===
        else:
            # Conservative: CRSI extremes only
            if crsi_extreme_oversold and trend_bullish:
                desired_signal = REDUCED_SIZE
            if crsi_extreme_overbought and trend_bearish and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 1.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 1.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 1.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish and CRSI not extreme overbought
                if trend_bullish and crsi_4h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend still bearish and CRSI not extreme oversold
                if trend_bearish and crsi_4h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses + CRSI overbought
            if trend_bearish and crsi_4h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses + CRSI oversold
            if trend_bullish and crsi_4h[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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