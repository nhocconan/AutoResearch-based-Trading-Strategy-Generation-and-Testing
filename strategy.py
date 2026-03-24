#!/usr/bin/env python3
"""
Experiment #612: 12h Primary + 1d HTF — Regime-Adaptive Connors RSI + HMA Trend

Hypothesis: 12h timeframe with Connors RSI for mean reversion entries combined with
1d HMA trend bias and Choppiness regime filter will generate consistent trades with
positive Sharpe across all symbols. Connors RSI (RSI-3 + RSI-Streak + PercentRank) / 3
has 75% win rate for mean reversion. Combined with HTF trend filter, this should
work in both bull and bear markets.

Key improvements over failed experiments:
1. LOOSEN entry conditions - RSI(3) < 30 or > 70 (not extreme 10/90)
2. Connors RSI simplified for faster signals
3. Choppiness regime switch: range=mean revert, trend=trend follow
4. 1d HMA(21) for trend bias - only trade with HTF direction
5. Discrete signals: 0.0, ±0.25, ±0.30 to minimize fee churn
6. ATR(14)*2.5 stoploss on all positions

Strategy logic:
1. 1d HMA(21) = trend bias (price > HMA = bull, price < HMA = bear)
2. 12h Choppiness(14) = regime (CHOP>55 = range, CHOP<45 = trend)
3. 12h Connors RSI(3,2,100) = entry timing
4. 12h ATR(14) = position sizing and stoploss

Entry rules (LOOSENED for trade generation):
- RANGE (CHOP>55): Long when CRSI<25 + price>1d_HMA, Short when CRSI>75 + price<1d_HMA
- TREND (CHOP<45): Long when price>1d_HMA + pullback to HMA(21), Short when price<1d_HMA + rally to HMA(21)

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_hma_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak Component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        streak = 0
        for j in range(period):
            idx = i - j
            if idx <= 0:
                break
            if close[idx] > close[idx - 1]:
                streak += 1
            elif close[idx] < close[idx - 1]:
                streak -= 1
        
        # Convert streak to 0-100 scale
        # Max streak over period days = period, min = -period
        streak_rsi[i] = 50.0 + (streak / period) * 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank Component of Connors RSI
    Measures current price change vs past period changes
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period, n):
        current_change = close[i] - close[i - 1] if i > 0 else 0
        count_higher = 0
        
        for j in range(1, period):
            idx = i - j
            if idx <= 0:
                break
            past_change = close[idx] - close[idx - 1]
            if current_change > past_change:
                count_higher += 1
        
        pr[i] = (count_higher / (period - 1)) * 100.0
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100, extremes <10 or >90 signal mean reversion
    """
    rsi = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi + streak_rsi + pr) / 3.0
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
    
    atr = calculate_atr(high, low, close, period)
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=21)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=50)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d trend) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0   # Range-bound (mean revert)
        chop_trend = chop[i] < 45.0   # Trending (trend follow)
        
        # === CONNORS RSI EXTREMES (LOOSENED for trade generation) ===
        crsi_oversold = crsi[i] < 30.0
        crsi_overbought = crsi[i] > 70.0
        crsi_extreme_oversold = crsi[i] < 20.0
        crsi_extreme_overbought = crsi[i] > 80.0
        
        # === REGIME DETECTION ===
        is_range_regime = chop_range
        is_trend_regime = chop_trend
        
        # === ENTRY LOGIC (LOOSENED CONDITIONS) ===
        desired_signal = 0.0
        
        # RANGE REGIME: Mean reversion at CRSI extremes
        if is_range_regime:
            # Long: CRSI oversold + HTF bull bias
            if crsi_oversold and htf_bull:
                desired_signal = SIZE_BASE
            # Short: CRSI overbought + HTF bear bias
            elif crsi_overbought and htf_bear:
                desired_signal = -SIZE_BASE
            # Extreme reversal (trade against HTF for quick mean revert)
            elif crsi_extreme_oversold:
                desired_signal = SIZE_BASE * 0.8
            elif crsi_extreme_overbought:
                desired_signal = -SIZE_BASE * 0.8
        
        # TREND REGIME: Follow HTF direction on pullbacks
        elif is_trend_regime:
            # Long: HTF bull + pullback (CRSI not overbought)
            if htf_bull and hma_bull and crsi[i] < 60.0:
                desired_signal = SIZE_STRONG
            # Short: HTF bear + rally (CRSI not oversold)
            elif htf_bear and hma_bear and crsi[i] > 40.0:
                desired_signal = -SIZE_STRONG
            # HMA crossover confirmation
            elif htf_bull and close[i] > hma_12h[i] and close[i-1] <= hma_12h[i-1] if i > 0 else False:
                desired_signal = SIZE_BASE
            elif htf_bear and close[i] < hma_12h[i] and close[i-1] >= hma_12h[i-1] if i > 0 else False:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL/TRANSITION: Reduced signals
        else:
            # Only take extreme CRSI signals
            if crsi_extreme_oversold and htf_bull:
                desired_signal = SIZE_BASE * 0.6
            elif crsi_extreme_overbought and htf_bear:
                desired_signal = -SIZE_BASE * 0.6
        
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