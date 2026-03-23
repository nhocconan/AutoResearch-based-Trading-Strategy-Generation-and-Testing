#!/usr/bin/env python3
"""
Experiment #968: 30m Primary + 4h/1d HTF — Fisher Transform + Connors RSI + Session Filter

Hypothesis: After 667 failed strategies, combining Ehlers Fisher Transform (reversal detection)
with Connors RSI (mean reversion) and strict session/volume filters should work on 30m
while keeping trade count low (30-80/year) to minimize fee drag.

Key insights from research:
1. Fisher Transform (period=9): Catches reversals in bear markets better than RSI
   Long when Fisher crosses above -1.5, short when crosses below +1.5
2. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Entry when CRSI < 15 (long) or > 85 (short) within HTF trend
3. Session Filter: Only trade 8-20 UTC (highest volume, lowest slippage)
4. Volume Filter: volume > 0.8x 20-bar average (confirms move)
5. 4h HMA21 for medium-term trend bias
6. 1d HMA21 for macro regime filter

Why 30m timeframe with strict filters:
- Target 30-80 trades/year (use 3+ confluence to limit entries)
- HTF (4h/1d) provides signal DIRECTION
- 30m Fisher provides ENTRY TIMING precision
- Session filter cuts overnight noise (50%+ of bars excluded)

Critical improvements over failed 30m strategies:
- Session filter (8-20 UTC) reduces trades by ~60%
- Volume confirmation avoids false breakouts
- Fisher + CRSI confluence (both must agree)
- Discrete signal sizes (0.0, ±0.20, ±0.30) minimize fee churn
- ALL symbols MUST have positive Sharpe (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 40-70 trades/year with filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_crsi_session_4h1d_hma_volume_v1"
timeframe = "30m"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - detects price reversals.
    Normalizes price to -1 to +1 range, crossings signal reversals.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    # Calculate median price
    median = (high + low) / 2
    
    for i in range(period - 1, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            fisher_prev[i] = 0.0
            continue
        
        # Normalize price to 0-1 range
        x = (median[i] - lowest) / (highest - lowest)
        
        # Constrain to 0.001-0.999 to avoid log(0)
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Previous value for crossover detection
        if i > period - 1:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. < 10 = oversold, > 90 = overbought.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_pad = np.concatenate([[0], gain])
    loss_pad = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain_pad).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss_pad).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = np.clip(rsi_3, 0, 100)
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        up_streak = max(0, streak[i])
        down_streak = abs(min(0, streak[i]))
        
        if up_streak + down_streak > 0:
            streak_rsi[i] = 100 * up_streak / (up_streak + down_streak + 1e-10)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100) - where current close ranks in last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period - 1, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / (rank_period - 1) * 100
        percent_rank[i] = rank
    
    # Combine into Connors RSI
    for i in range(rank_period - 1, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    return crsi

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume / average volume."""
    n = len(volume)
    ratio = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        avg_vol = np.mean(volume[i-period+1:i+1])
        if avg_vol > 1e-10:
            ratio[i] = volume[i] / avg_vol
        else:
            ratio[i] = 1.0
    
    return ratio

def extract_hour_from_open_time(open_time_series):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time_series // (1000 * 60 * 60)) % 24
    return hours.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    fisher_30m, fisher_prev_30m = calculate_fisher_transform(high, low, close, period=9)
    crsi_30m = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_ratio_30m = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Extract UTC hour for session filter
    utc_hours = extract_hour_from_open_time(open_time)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(fisher_30m[i]) or np.isnan(fisher_prev_30m[i]):
            continue
        if np.isnan(crsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(vol_ratio_30m[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_confirmed = vol_ratio_30m[i] > 0.8
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long_cross = (fisher_30m[i] > -1.5) and (fisher_prev_30m[i] <= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_short_cross = (fisher_30m[i] < 1.5) and (fisher_prev_30m[i] >= 1.5)
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_30m[i] < 15
        crsi_overbought = crsi_30m[i] > 85
        crsi_extreme_oversold = crsi_30m[i] < 10
        crsi_extreme_overbought = crsi_30m[i] > 90
        
        desired_signal = 0.0
        
        # === LONG ENTRY (3+ confluence required) ===
        # Must have: session + volume + (HTF trend OR extreme CRSI) + Fisher cross
        if in_session and volume_confirmed:
            # Primary long: Bullish HTF + Fisher cross + CRSI oversold
            if (macro_bull or trend_4h_bullish) and fisher_long_cross and crsi_oversold:
                desired_signal = BASE_SIZE
            # Secondary long: Extreme CRSI + Fisher cross (stronger mean reversion)
            elif crsi_extreme_oversold and fisher_long_cross:
                desired_signal = BASE_SIZE
            # Tertiary long: HTF bullish + Fisher cross (trend following)
            elif (macro_bull and trend_4h_bullish) and fisher_long_cross:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY (3+ confluence required) ===
        if in_session and volume_confirmed:
            # Primary short: Bearish HTF + Fisher cross + CRSI overbought
            if (macro_bear or trend_4h_bearish) and fisher_short_cross and crsi_overbought:
                desired_signal = -BASE_SIZE
            # Secondary short: Extreme CRSI + Fisher cross (stronger mean reversion)
            elif crsi_extreme_overbought and fisher_short_cross:
                desired_signal = -BASE_SIZE
            # Tertiary short: HTF bearish + Fisher cross (trend following)
            elif (macro_bear and trend_4h_bearish) and fisher_short_cross:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF trend intact and CRSI not overbought
                if (macro_bull or trend_4h_bullish) and crsi_30m[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HTF trend intact and CRSI not oversold
                if (macro_bear or trend_4h_bearish) and crsi_30m[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF trend reverses + CRSI overbought
            if macro_bear and trend_4h_bearish and crsi_30m[i] > 80:
                desired_signal = 0.0
            # Exit if Fisher flips strongly short
            if fisher_30m[i] < -0.5 and fisher_prev_30m[i] > 0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF trend reverses + CRSI oversold
            if macro_bull and trend_4h_bullish and crsi_30m[i] < 20:
                desired_signal = 0.0
            # Exit if Fisher flips strongly long
            if fisher_30m[i] > 0.5 and fisher_prev_30m[i] < 0:
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
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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