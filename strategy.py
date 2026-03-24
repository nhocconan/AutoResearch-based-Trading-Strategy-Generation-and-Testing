#!/usr/bin/env python3
"""
Experiment #860: 6h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: 6h timeframe is underexplored middle-ground between 4h and 12h.
Fisher Transform excels at catching reversals in bear/range markets (proven edge).
Combined with Choppiness Index regime filter and 1d HMA trend bias, this should
outperform pure trend strategies that failed on BTC/ETH during 2022 crash.

Key innovations:
1. Ehlers Fisher Transform(9) - catches reversals better than RSI in bear markets
2. 1d HMA(21) for HTF trend bias - smoother than EMA, less lag
3. 1w HMA(21) for macro regime weight - adds conviction filter
4. 6h Choppiness(14) regime switch - trend vs mean-revert logic
5. Volume confirmation filter - reduces false breakouts
6. Asymmetric sizing: 0.25 base, 0.35 strong (with HTF confluence)
7. ATR(14) 2.5x trailing stop for risk management

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- TREND REGIME (CHOP<50): Fisher crosses -1.5 + 1d HMA bull → LONG
- TREND REGIME (CHOP<50): Fisher crosses +1.5 + 1d HMA bear → SHORT
- RANGE REGIME (CHOP>=50): Fisher < -2.0 → LONG, Fisher > +2.0 → SHORT
- Volume > 0.8 * SMA20(volume) required for all entries

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.35 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    
    Steps:
    1. Calculate typical price: (High + Low) / 2
    2. Normalize to -1 to +1 range
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    
    Long signal: Fisher crosses above -1.5 from below
    Short signal: Fisher crosses below +1.5 from above
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    # Normalize to -1 to +1 using highest high and lowest low over period
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize: map price to -0.99 to +0.99 range
        normalized = 2.0 * (typical[i] - lowest) / price_range - 1.0
        normalized = max(-0.99, min(0.99, normalized))  # Clamp to avoid ln(0)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

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
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 50 as threshold for regime switch
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(vol_sma_20[i]):
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
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === MACRO BIAS (1w HMA) - adds conviction ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = False
        fisher_cross_down = False
        
        if not np.isnan(fisher_prev[i]) and not np.isnan(fisher[i]):
            # Cross above -1.5 (long signal)
            fisher_cross_up = (fisher_prev[i] <= -1.5) and (fisher[i] > -1.5)
            # Cross below +1.5 (short signal)
            fisher_cross_down = (fisher_prev[i] >= 1.5) and (fisher[i] < 1.5)
        
        # Fisher extreme oversold/overbought
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0
        chop_ranging = chop_14[i] >= 50.0
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        if htf_1d_bull:
            # Bullish HTF bias - prefer longs
            if chop_trending:
                # Trend regime: Fisher cross up + volume
                if fisher_cross_up and volume_ok:
                    # Check 1w for conviction
                    if htf_1w_bull:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            else:
                # Range regime: Fisher extreme low
                if fisher_extreme_low and volume_ok:
                    if htf_1w_bull:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
        
        elif htf_1d_bear:
            # Bearish HTF bias - prefer shorts
            if chop_trending:
                # Trend regime: Fisher cross down + volume
                if fisher_cross_down and volume_ok:
                    if htf_1w_bear:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
            else:
                # Range regime: Fisher extreme high
                if fisher_extreme_high and volume_ok:
                    if htf_1w_bear:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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