#!/usr/bin/env python3
"""
Experiment #1512: 12h Primary + 1d/1w HTF — Dual Regime (Chop/Trend) + Connors RSI

Hypothesis: Based on #1506 success (Sharpe=0.138) but need to beat current best (Sharpe=0.618).
Key insight from research: Choppiness Index + Connors RSI achieved ETH Sharpe +0.923 in tests.

Design:
1. Use Choppiness Index (CHOP) to detect regime: CHOP>61.8 = range, CHOP<38.2 = trend
2. In TREND regime: Use 1w/1d HMA for direction + Donchian breakout for entry
3. In RANGE regime: Use Connors RSI for mean reversion (CRSI<10 long, >90 short)
4. Position size adapts: 0.30 in trend, 0.20 in chop (less conviction in ranges)
5. ATR 2.5x trailing stop on all positions
6. 1w HMA as ultimate trend filter (only long if price>1w HMA, only short if below)

Timeframe: 12h (as required)
HTF: 1d + 1w (dual HTF for stronger trend confirmation)
Position Size: 0.20-0.30 (discrete levels)
Target: 30-60 trades/train, 8-15 trades/test, Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_donchian_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Better for mean reversion than standard RSI
    """
    n = len(close)
    if n < period_rank:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period_rsi)
    
    # RSI Streak: RSI of consecutive up/down days
    direction = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    for i in range(1, n):
        if direction[i] > 0:
            streak[i] = streak[i-1] + 1 if direction[i-1] >= 0 else 1
        elif direction[i] < 0:
            streak[i] = streak[i-1] - 1 if direction[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan)
    for i in range(period_streak, n):
        if np.isnan(streak[i]):
            continue
        # Map streak to 0-100 scale
        if streak[i] > 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Percent Rank: where current close ranks in last 100 closes
    percent_rank = np.full(n, np.nan)
    for i in range(period_rank, n):
        window = close[i-period_rank+1:i+1]
        if np.any(np.isnan(window)):
            continue
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = (count_below / (period_rank - 1)) * 100
    
    # Combine
    crsi = np.full(n, np.nan)
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures if market is trending or ranging
    CHOP > 61.8 = ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel: highest high and lowest low over period"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend (ultimate filter)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # Larger size in trending regime
    SIZE_CHOP = 0.20   # Smaller size in ranging regime
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_12h[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop[i] < 45.0  # Trending market
        is_chop_regime = chop[i] > 55.0   # Ranging market
        # Neutral zone: 45-55, use trend logic but smaller size
        
        # === MACRO TREND FILTERS ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        h12_bull = close[i] > hma_12h[i]
        h12_bear = close[i] < hma_12h[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] >= donchian_upper[i] * 0.998  # Near upper band
        donchian_breakout_short = close[i] <= donchian_lower[i] * 1.002  # Near lower band
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        position_size = SIZE_TREND if is_trend_regime else SIZE_CHOP
        
        if is_trend_regime:
            # TREND FOLLOWING LOGIC
            # Long: Weekly bull + Daily bull + 12h bull + Donchian breakout
            if weekly_bull and daily_bull and h12_bull and donchian_breakout_long:
                desired_signal = SIZE_TREND
            # Long: Weekly bull + Daily bull + 12h above HMA + RSI not overbought
            elif weekly_bull and daily_bull and h12_bull and rsi[i] < 65.0:
                desired_signal = SIZE_TREND * 0.8
            # Long: Weekly bull + 12h bull (looser for more trades)
            elif weekly_bull and h12_bull and rsi[i] < 60.0:
                desired_signal = SIZE_TREND * 0.6
            
            # Short: Weekly bear + Daily bear + 12h bear + Donchian breakout
            elif weekly_bear and daily_bear and h12_bear and donchian_breakout_short:
                desired_signal = -SIZE_TREND
            # Short: Weekly bear + Daily bear + 12h below HMA + RSI not oversold
            elif weekly_bear and daily_bear and h12_bear and rsi[i] > 35.0:
                desired_signal = -SIZE_TREND * 0.8
            # Short: Weekly bear + 12h bear (looser for more trades)
            elif weekly_bear and h12_bear and rsi[i] > 40.0:
                desired_signal = -SIZE_TREND * 0.6
        
        elif is_chop_regime:
            # MEAN REVERSION LOGIC (Connors RSI)
            # Long: CRSI < 15 (oversold) + Weekly bull (only long in bull macro)
            if crsi[i] < 15.0 and weekly_bull:
                desired_signal = SIZE_CHOP
            # Long: CRSI < 20 + price near Donchian lower
            elif crsi[i] < 20.0 and close[i] < donchian_lower[i] * 1.02:
                desired_signal = SIZE_CHOP * 0.8
            # Long: CRSI < 25 + RSI < 35 (deep pullback)
            elif crsi[i] < 25.0 and rsi[i] < 35.0 and weekly_bull:
                desired_signal = SIZE_CHOP * 0.6
            
            # Short: CRSI > 85 (overbought) + Weekly bear (only short in bear macro)
            elif crsi[i] > 85.0 and weekly_bear:
                desired_signal = -SIZE_CHOP
            # Short: CRSI > 80 + price near Donchian upper
            elif crsi[i] > 80.0 and close[i] > donchian_upper[i] * 0.98:
                desired_signal = -SIZE_CHOP * 0.8
            # Short: CRSI > 75 + RSI > 65 (strong rally)
            elif crsi[i] > 75.0 and rsi[i] > 65.0 and weekly_bear:
                desired_signal = -SIZE_CHOP * 0.6
        
        else:
            # NEUTRAL ZONE: Use trend logic but smaller size
            if weekly_bull and h12_bull and rsi[i] < 55.0:
                desired_signal = SIZE_CHOP * 0.8
            elif weekly_bear and h12_bear and rsi[i] > 45.0:
                desired_signal = -SIZE_CHOP * 0.8
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal >= SIZE_TREND * 0.7:
            final_signal = SIZE_TREND * 0.8
        elif desired_signal >= SIZE_TREND * 0.5:
            final_signal = SIZE_CHOP
        elif desired_signal >= SIZE_CHOP * 0.7:
            final_signal = SIZE_CHOP * 0.8
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.7:
            final_signal = -SIZE_TREND * 0.8
        elif desired_signal <= -SIZE_TREND * 0.5:
            final_signal = -SIZE_CHOP
        elif desired_signal <= -SIZE_CHOP * 0.7:
            final_signal = -SIZE_CHOP * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals