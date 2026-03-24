#!/usr/bin/env python3
"""
Experiment #1490: 1h Primary + 4h/12h HTF — Regime-Adaptive Strategy

Hypothesis: After 1100+ failed strategies, the pattern is clear:
1. Pure trend-following fails in bear/range markets (2022 crash, 2025 bear)
2. Pure mean-reversion fails in strong trends
3. REGIME-ADAPTIVE approach should work: detect market state, adapt logic

Key insight: Choppiness Index (CHOP) distinguishes range vs trend regimes.
- CHOP > 55 = Range → Mean revert with Connors RSI extremes
- CHOP < 45 = Trend → Follow HTF trend with pullback entries
- CHOP 45-55 = Transition → Stay flat or reduce size

This strategy uses:
- 12h HMA for macro trend bias (call get_htf_data ONCE!)
- 4h HMA for intermediate trend direction
- 1h Choppiness Index for regime detection
- 1h Connors RSI for mean reversion entries
- Session filter (8-20 UTC) for liquidity
- Volume filter (>0.7x avg) for confirmation
- ATR(14)*2.5 trailing stoploss

Why 1h + 4h/12h should work:
1. 1h = target 30-60 trades/year (fee drag ~1.5-3%)
2. 12h/4h HMA filters prevent trading against macro trend
3. Regime-adaptive logic works in both bull AND bear markets
4. Connors RSI (3-period) catches quick reversals in ranges
5. Discrete signal sizes (0.0, ±0.20, ±0.25) minimize fee churn

Timeframe: 1h
HTF: 4h, 12h (call get_htf_data ONCE before loop!)
Position Size: 0.25 (smaller for 1h TF)
Target: 30-60 trades/year, Sharpe > 0.618, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_chop_crsi_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            if np.all(~np.isnan(series[i - span + 1:i + 1])):
                result[i] = np.sum(series[i - span + 1:i + 1] * weights) / np.sum(weights)
        return result
    
    close_series = pd.Series(close)
    wma_half = wma(close, period // 2)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    sqrt_period = int(np.sqrt(period))
    
    for i in range(period, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2 * wma_half[i] - wma_full[i]
            # Apply WMA on the difference with sqrt(period)
            if i >= sqrt_period - 1:
                weights = np.arange(1, sqrt_period + 1)
                start_idx = i - sqrt_period + 1
                if not np.isnan(diff) and np.all(~np.isnan([2*wma_half[j] - wma_full[j] for j in range(start_idx, i+1) if not np.isnan(wma_half[j]) and not np.isnan(wma_full[j])])):
                    vals = []
                    for j in range(start_idx, i + 1):
                        if not np.isnan(wma_half[j]) and not np.isnan(wma_full[j]):
                            vals.append(2 * wma_half[j] - wma_full[j])
                        else:
                            vals.append(np.nan)
                    if len(vals) == sqrt_period and not np.any(np.isnan(vals)):
                        hma[i] = np.sum(np.array(vals) * weights) / np.sum(weights)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures if market is trending or ranging
    CHOP = 100 * LOG10(SUM(ATR, n) / (Max High - Min Low)) / LOG10(n)
    CHOP > 61.8 = Range, CHOP < 38.2 = Trend
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
        sum_atr = np.nansum(tr[i - period + 1:i + 1])
        highest_high = np.nanmax(high[i - period + 1:i + 1])
        lowest_low = np.nanmin(low[i - period + 1:i + 1])
        
        if highest_low := (highest_high - lowest_low) > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Excellent for mean reversion entries
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi_short[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi_short[loss_smooth <= 1e-10] = 100.0
    rsi_short[:rsi_period] = np.nan
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak values (period=2)
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rsi_mask = streak_loss_smooth > 1e-10
    streak_rsi[streak_rsi_mask] = 100.0 - (100.0 / (1.0 + streak_gain_smooth[streak_rsi_mask] / streak_loss_smooth[streak_rsi_mask]))
    streak_rsi[~streak_rsi_mask] = 100.0
    streak_rsi[:streak_period] = np.nan
    
    # Percent Rank (100) - where does current return rank vs last 100 bars
    percent_rank = np.full(n, np.nan)
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i]
        current = returns[i]
        if len(window) > 0:
            count_below = np.sum(window < current)
            percent_rank[i] = 100.0 * count_below / len(window)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan)
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

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

def calculate_volume_avg(volume, period=20):
    """Average volume for volume filter"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    # Calculate 1h HMA for additional trend filter
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h TF
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_range = chop[i] > 55.0  # Range market
        is_trend = chop[i] < 45.0  # Trending market
        is_transition = 45.0 <= chop[i] <= 55.0  # Transition
        
        # === MACRO TREND (12h HMA) ===
        daily_bull = close[i] > hma_12h_aligned[i]
        daily_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === SHORT-TERM TREND (1h HMA) ===
        hma_1h_bull = close[i] > hma_1h[i]
        hma_1h_bear = close[i] < hma_1h[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Strong mean reversion long signal
        crsi_overbought = crsi[i] > 85.0  # Strong mean reversion short signal
        crsi_mild_oversold = crsi[i] < 25.0
        crsi_mild_overbought = crsi[i] > 75.0
        
        # === DESIRED SIGNAL - REGIME ADAPTIVE ===
        desired_signal = 0.0
        
        if is_range and in_session and volume_ok:
            # MEAN REVERSION in range market
            if crsi_oversold and daily_bull:  # Long in uptrend range
                desired_signal = BASE_SIZE
            elif crsi_overbought and daily_bear:  # Short in downtrend range
                desired_signal = -BASE_SIZE
            elif crsi_mild_oversold and hma_4h_bull:  # Weaker long
                desired_signal = BASE_SIZE * 0.6
            elif crsi_mild_overbought and hma_4h_bear:  # Weaker short
                desired_signal = -BASE_SIZE * 0.6
        
        elif is_trend and in_session and volume_ok:
            # TREND FOLLOWING with pullback entries
            if daily_bull and hma_4h_bull and hma_1h_bear and crsi_mild_oversold:
                # Pullback long in uptrend
                desired_signal = BASE_SIZE
            elif daily_bear and hma_4h_bear and hma_1h_bull and crsi_mild_overbought:
                # Pullback short in downtrend
                desired_signal = -BASE_SIZE
            elif daily_bull and hma_4h_bull and hma_1h_bull:
                # Strong trend continuation
                desired_signal = BASE_SIZE * 0.7
            elif daily_bear and hma_4h_bear and hma_1h_bear:
                # Strong trend continuation
                desired_signal = -BASE_SIZE * 0.7
        
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.6
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