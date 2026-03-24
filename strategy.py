#!/usr/bin/env python3
"""
Experiment #526: 1d Primary + 1w HTF — Dual Regime: Connors RSI Mean Reversion + Donchian Trend

Hypothesis: Daily timeframe with weekly HTF bias provides optimal balance between
trade frequency (20-50/year) and signal quality. Dual regime approach:
1. RANGE regime (CHOP>55): Connors RSI mean reversion at extremes
2. TREND regime (CHOP<45): Donchian breakout with HMA confirmation

Key improvements over failed experiments:
1. Connors RSI (CRSI) proven 75% win rate on mean reversion
2. Choppiness Index cleanly separates regime (no ADX complexity)
3. 1w HMA(21) for macro bias - only trade with weekly trend
4. ATR(14)*2.5 stoploss on all positions
5. Discrete signal sizes (0.0, ±0.25, ±0.30) to minimize fee churn

Strategy logic:
1. 1w HMA(21) = macro trend bias (long only if price>1w_HMA, short if price<1w_HMA)
2. 1d Choppiness(14) = regime detection (CHOP>55=range, CHOP<45=trend)
3. 1d Connors RSI = mean reversion entry (CRSI<10 long, CRSI>90 short)
4. 1d Donchian(20) = trend breakout entry (break high/low)
5. 1d ATR(14) = position sizing and stoploss

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    RSI(3): Fast RSI for short-term extremes
    RSI_Streak(2): RSI of consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 days
    
    CRSI < 10 = extreme oversold (long)
    CRSI > 90 = extreme overbought (short)
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_close = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_close[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_close[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI of streak (consecutive up/down days)
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
        if avg_streak_loss[i] < 1e-10 and avg_streak_gain[i] > 0:
            rsi_streak[i] = 100.0
        elif avg_streak_loss[i] < 1e-10:
            rsi_streak[i] = 50.0
        else:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank (where current close ranks vs last rank_period days)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank = 100.0 * count_below / (rank_period - 1)
        
        # Average the three components
        crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range-bound (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
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

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel - breakout levels
    Upper = highest high of last period bars
    Lower = lowest low of last period bars
    """
    n = len(close)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
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
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF MACRO BIAS (1w HMA) ===
        htf_bull = close[i] > hma_1w_aligned[i]
        htf_bear = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_regime = chop[i] > 55.0   # Mean reversion
        is_trend_regime = chop[i] < 45.0   # Trend following
        is_transition = not is_range_regime and not is_trend_regime
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # RANGE REGIME: Connors RSI mean reversion
        if is_range_regime:
            # Long: CRSI extreme oversold + HTF bull bias
            if crsi_extreme_oversold and htf_bull:
                desired_signal = SIZE_STRONG
            elif crsi_oversold and htf_bull:
                desired_signal = SIZE_BASE
            # Short: CRSI extreme overbought + HTF bear bias
            elif crsi_extreme_overbought and htf_bear:
                desired_signal = -SIZE_STRONG
            elif crsi_overbought and htf_bear:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Donchian breakout with HTF confirmation
        elif is_trend_regime:
            # Long breakout + HTF bull
            if donchian_breakout_long and htf_bull:
                desired_signal = SIZE_STRONG
            # Short breakout + HTF bear
            elif donchian_breakout_short and htf_bear:
                desired_signal = -SIZE_STRONG
        
        # TRANSITION REGIME: Wait for confirmation or reduce size
        elif is_transition:
            # Only take extreme CRSI signals in transition
            if crsi_extreme_oversold and htf_bull:
                desired_signal = SIZE_BASE * 0.8
            elif crsi_extreme_overbought and htf_bear:
                desired_signal = -SIZE_BASE * 0.8
        
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
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
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