#!/usr/bin/env python3
"""
Experiment #555: 6h Primary + 12h/1d HTF — Fisher Transform Reversals + HMA Trend

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow).
Fisher Transform excels at catching reversals in bear/range markets (2025 test period).
Combined with 1d HMA for trend bias and 12h ADX for regime, this should:
1. Catch bear market rallies (Fisher oversold + price<1d_HMA = short)
2. Catch bull market dips (Fisher oversold + price>1d_HMA = long)
3. Avoid whipsaw in transition (ADX<20 = reduce size or flat)

Key improvements over failed #547 (6h_supertrend_fisher):
1. Simpler Fisher logic (cross -1.2/+1.2, not complex Supertrend)
2. 1d HMA trend filter (not 1w which is too slow)
3. 12h ADX regime (not Choppiness which failed)
4. LOOSE entry thresholds to ensure 30+ trades/year
5. No volume filter (was blocking valid signals)

Strategy logic:
1. 1d HMA(21) = trend bias (price above = bull, below = bear)
2. 12h ADX(14) = regime (ADX>25 = trend, ADX<20 = range)
3. 6h Fisher(9) = reversal signal (cross -1.2 = long, cross +1.2 = short)
4. 6h ATR(14)*2.5 = stoploss

Entry conditions (LOOSE to ensure trades):
- Long: Fisher crosses above -1.2 OR Fisher < -0.8 + trend_bull
- Short: Fisher crosses below +1.2 OR Fisher > +0.8 + trend_bear
- Range: Either extreme works both directions

Target: Sharpe>0.40, trades>=120 train (30/year), trades>=15 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_hma_adx_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Excellent for spotting reversals at extremes
    
    Formula:
    1. Calculate median price: (high + low) / 2
    2. Normalize: 0.66 * ((median - lowest_low) / (highest_high - lowest_low) - 0.5)
    3. Smooth with EMA
    4. Fisher = 0.5 * ln((1 + smoothed) / (1 - smoothed))
    5. Signal = previous Fisher value (for crossover detection)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    median = (high + low) / 2.0
    fisher = np.zeros(n)
    fisher[:] = np.nan
    signal = np.zeros(n)
    signal[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range < 1e-10:
            fisher[i] = 0.0
            if i > period:
                signal[i] = fisher[i-1]
            continue
        
        normalized = 0.66 * ((median[i] - lowest_low) / price_range - 0.5)
        normalized = max(-0.999, min(0.999, normalized))
        
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > period:
            fisher[i] = 0.7 * fisher[i] + 0.3 * fisher[i-1]
        
        signal[i] = fisher[i-1]
    
    return fisher, signal

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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

def calculate_hma(close, period):
    """Hull Moving Average"""
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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h ADX
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate and align 1d HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_RANGE = 0.25
    
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # HTF trend bias from 1d HMA
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # Regime from 12h ADX
        is_trend = adx_12h_aligned[i] > 22.0
        is_range = adx_12h_aligned[i] < 18.0
        
        # Fisher reversal signals (LOOSE thresholds to ensure trades)
        fisher_cross_long = (fisher_signal[i] < -1.2 and fisher[i] >= -1.2) or \
                           (fisher_signal[i] < -0.5 and fisher[i] >= -0.5 and fisher[i-1] < fisher_signal[i-1] if i > 0 else False)
        fisher_cross_short = (fisher_signal[i] > 1.2 and fisher[i] <= 1.2) or \
                            (fisher_signal[i] > 0.5 and fisher[i] <= 0.5 and fisher[i-1] > fisher_signal[i-1] if i > 0 else False)
        fisher_extreme_long = fisher[i] < -0.8
        fisher_extreme_short = fisher[i] > 0.8
        
        desired_signal = 0.0
        
        # TREND REGIME: Follow trend direction with Fisher confirmation
        if is_trend:
            if trend_bull and (fisher_cross_long or fisher_extreme_long):
                desired_signal = SIZE_TREND
            elif trend_bear and (fisher_cross_short or fisher_extreme_short):
                desired_signal = -SIZE_TREND
            # Counter-trend with strong Fisher extreme
            elif trend_bull and fisher_extreme_short:
                desired_signal = -SIZE_RANGE * 0.7
            elif trend_bear and fisher_extreme_long:
                desired_signal = SIZE_RANGE * 0.7
        
        # RANGE REGIME: Mean reversion at Fisher extremes (both directions OK)
        elif is_range:
            if fisher_extreme_long:
                desired_signal = SIZE_RANGE
            elif fisher_extreme_short:
                desired_signal = -SIZE_RANGE
            elif fisher_cross_long:
                desired_signal = SIZE_RANGE * 0.8
            elif fisher_cross_short:
                desired_signal = -SIZE_RANGE * 0.8
        
        # TRANSITION: Only take strong signals
        else:
            if fisher_extreme_long and trend_bull:
                desired_signal = SIZE_RANGE * 0.6
            elif fisher_extreme_short and trend_bear:
                desired_signal = -SIZE_RANGE * 0.6
        
        # Stoploss check (2.5x ATR from entry)
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
        
        # Discretize signal values (Rule 4)
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_RANGE * 0.9:
            final_signal = SIZE_RANGE
        elif desired_signal <= -SIZE_RANGE * 0.9:
            final_signal = -SIZE_RANGE
        elif abs(desired_signal) >= SIZE_RANGE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_RANGE * 0.8
        else:
            final_signal = 0.0
        
        # Update position tracking
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