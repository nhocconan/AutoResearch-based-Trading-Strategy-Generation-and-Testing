#!/usr/bin/env python3
"""
Experiment #596: 30m Primary + 4h/1d HTF — Choppiness + cRSI + Session Filter

Hypothesis: 30m timeframe with strict confluence filters generates optimal trade count
(40-80/year) while maintaining quality. Key insight from 500+ failed experiments:
- Lower TF needs HTF direction filter to avoid whipsaw
- Session filter (08-20 UTC) reduces overnight noise and trade count
- cRSI (Connors RSI) superior to standard RSI for mean reversion timing
- Choppiness Index filters regime - only trade mean reversion in range markets

CRITICAL LESSON: Previous 30m/15m strategies generated 0 trades (Sharpe=0.000)
because entry conditions were TOO STRICT. This strategy uses LOOSER thresholds
to ensure trade generation while maintaining quality via HTF + session filters.

Strategy logic:
1. 1d HMA(21) = macro trend bias (slow filter)
2. 4h HMA(21) = medium trend direction
3. 30m Choppiness(14) = regime (CHOP>55 = range, only mean revert in range)
4. 30m cRSI = entry timing (cRSI<20 long, cRSI>80 short)
5. Session filter = 08-20 UTC only (reduces trades by ~40%)
6. ATR(14)*2.5 stoploss on all positions

Entry conditions (LOOSER to ensure trades):
- Long: 4h_HMA_bull OR 1d_HMA_bull + CHOP>50 + cRSI<25 + session
- Short: 4h_HMA_bear OR 1d_HMA_bear + CHOP>50 + cRSI>75 + session

Target: Sharpe>0.40, trades>=40/year train, trades>=5/test per symbol
Timeframe: 30m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Short-term momentum
    RSI(Streak): Duration of consecutive up/down days
    PercentRank: Where current return ranks vs last 100 periods
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) - very short term
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI(Streak) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    for i in range(streak_period, n):
        if avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # PercentRank(100) - where current return ranks vs last 100
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0.0], returns])
    
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        count_below = np.sum(window < returns[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for medium trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = -0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Convert open_time (milliseconds) to hour
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === HTF BIAS (4h medium + 1d macro) ===
        htf_bull = close[i] > hma_4h_aligned[i] or close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i] or close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 50.0   # Range-bound (mean reversion valid)
        chop_trend = chop[i] < 45.0   # Trending
        
        # === cRSI EXTREMES (LOOSER thresholds for trade generation) ===
        crsi_oversold = crsi[i] < 25.0    # Long entry
        crsi_overbought = crsi[i] > 75.0  # Short entry
        crsi_extreme_os = crsi[i] < 15.0  # Strong long
        crsi_extreme_ob = crsi[i] > 85.0  # Strong short
        
        # === cRSI RECOVERY (additional entry trigger) ===
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 and not np.isnan(crsi[i-1]) else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 and not np.isnan(crsi[i-1]) else False
        
        # === ENTRY LOGIC (LOOSE conditions to ensure trades) ===
        desired_signal = 0.0
        
        # LONG entries - mean reversion in range market
        if in_session and chop_range:
            # Strong long: extreme cRSI + HTF bull bias
            if crsi_extreme_os and htf_bull:
                desired_signal = SIZE_LONG
            # Standard long: cRSI oversold + HTF neutral/bull
            elif crsi_oversold and (htf_bull or not htf_bear):
                desired_signal = SIZE_LONG * 0.8
            # cRSI recovery from oversold
            elif crsi[i] < 30.0 and crsi_rising and htf_bull:
                desired_signal = SIZE_LONG * 0.6
        
        # SHORT entries - mean reversion in range market
        elif in_session and chop_range:
            # Strong short: extreme cRSI + HTF bear bias
            if crsi_extreme_ob and htf_bear:
                desired_signal = SIZE_SHORT
            # Standard short: cRSI overbought + HTF neutral/bear
            elif crsi_overbought and (htf_bear or not htf_bull):
                desired_signal = SIZE_SHORT * 0.8
            # cRSI recovery from overbought
            elif crsi[i] > 70.0 and crsi_falling and htf_bear:
                desired_signal = SIZE_SHORT * 0.6
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_LONG * 0.9:
            final_signal = SIZE_LONG
        elif desired_signal <= SIZE_SHORT * 0.9:
            final_signal = SIZE_SHORT
        elif desired_signal >= SIZE_LONG * 0.5:
            final_signal = SIZE_LONG * 0.8
        elif desired_signal <= SIZE_SHORT * 0.5:
            final_signal = SIZE_SHORT * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals