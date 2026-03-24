#!/usr/bin/env python3
"""
Experiment #035: 6h Primary + 1d/1w HTF — Fisher Transform + Choppiness Regime + Dual HMA

Hypothesis: 6h timeframe needs a different approach than 4h/12h strategies that failed.
Key insight from 34 failed experiments:
- Pure trend (HMA/EMA/KAMA) fails on BTC/ETH in bear markets (2022 crash, 2025 bear)
- Pure mean-reversion (RSI/CRSI) fails on SOL during strong trends
- Volume filters add noise without edge on 6h timeframe

SOLUTION: Fisher Transform for reversals + Choppiness regime + Asymmetric HTF bias
- Fisher Transform (Ehlers): Catches sharp reversals better than RSI, works in bear rallies
- Choppiness Index: Switch between mean-revert (CHOP>55) and trend-follow (CHOP<45)
- 1d HMA(50): Intermediate trend bias
- 1w HMA(21): Major trend direction (very slow, avoids whipsaw)
- Asymmetric logic: Long bias when 1w HMA bull, short only on strong signals when 1w HMA bear
- ATR volatility filter: Skip entries when ATR ratio > 2.5 (panic spikes)

Why 6h specifically:
- 30-60 trades/year target (lower fee drag than 4h, more signals than 12h)
- Captures multi-day swings without 15m/1h noise
- 6h candles align well with funding rate cycles (8h)

Position sizing: 0.28 (28% of capital, conservative)
Stoploss: 2.5x ATR trailing
Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_hma_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_prev = np.zeros(n)
    fisher_prev[:] = np.nan
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        range_hl = highest - lowest
        
        if range_hl < 1e-10:
            fisher[i] = fisher[i-1] if i > period else 0.0
            fisher_prev[i] = fisher[i-1] if i > period else 0.0
            continue
        
        # Normalize price to 0-1 range
        price_norm = (hl2 - lowest) / range_hl
        
        # Clamp to avoid division issues
        price_norm = max(0.001, min(0.999, price_norm))
        
        # Fisher calculation
        fisher_val = 0.5 * np.log((1.0 + price_norm) / (1.0 - price_norm + 1e-10))
        
        # Smooth with previous value (Ehlers uses 0.67 weighting)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher_val + 0.33 * fisher[i-1]
        else:
            fisher[i] = fisher_val
        
        fisher_prev[i] = fisher[i-1] if i > period else fisher[i]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
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

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss and volatility filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(atr, period_short=7, period_long=30):
    """ATR ratio for volatility spike detection"""
    n = len(atr)
    if n < period_long:
        return np.full(n, np.nan)
    
    atr_short = pd.Series(atr).ewm(span=period_short, min_periods=period_short, adjust=False).mean().values
    atr_long = pd.Series(atr).ewm(span=period_long, min_periods=period_long, adjust=False).mean().values
    
    ratio = np.zeros(n)
    ratio[:] = np.nan
    for i in range(period_long, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(atr, period_short=7, period_long=30)
    
    # 6h HMA for local trend
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(hma_6h[i]):
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
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range/mean-revert regime
        is_trending = chop[i] < 45.0  # Trend regime
        is_neutral = not is_choppy and not is_trending
        
        # === VOLATILITY FILTER ===
        # Skip entries during extreme volatility spikes (panic)
        vol_spike = atr_ratio[i] > 2.5
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher extreme levels for mean-reversion
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === 6h HMA LOCAL TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === DESIRED SIGNAL (Asymmetric Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending and not vol_spike:
            # TREND REGIME: Follow Fisher reversals with HTF confirmation
            # LONG: Fisher reversal + 1w bull bias + 1d bull + 6h bull
            if fisher_long and htf_1w_bull and htf_1d_bull and hma_6h_bull:
                desired_signal = SIZE
            # LONG (weaker): Fisher reversal + 1w bull (ignore 1d if strong)
            elif fisher_long and htf_1w_bull and htf_1d_bull:
                desired_signal = SIZE * 0.7
            # SHORT: Fisher reversal + 1w bear bias + 1d bear + 6h bear (stricter for shorts)
            elif fisher_short and htf_1w_bear and htf_1d_bear and hma_6h_bear:
                desired_signal = -SIZE
            # SHORT (weaker): Fisher reversal + both HTF bear
            elif fisher_short and htf_1w_bear and htf_1d_bear:
                desired_signal = -SIZE * 0.7
        
        elif is_choppy and not vol_spike:
            # CHOPPY REGIME: Mean-revert at Fisher extremes
            # LONG: Fisher oversold + 1w not strongly bear
            if fisher_oversold and not htf_1w_bear:
                desired_signal = SIZE
            # LONG (weaker): Fisher oversold + 6h bull
            elif fisher_oversold and hma_6h_bull:
                desired_signal = SIZE * 0.7
            # SHORT: Fisher overbought + 1w not strongly bull
            if fisher_overbought and not htf_1w_bull:
                desired_signal = -SIZE
            # SHORT (weaker): Fisher overbought + 6h bear
            elif fisher_overbought and hma_6h_bear:
                desired_signal = -SIZE * 0.7
        
        elif is_neutral and not vol_spike:
            # NEUTRAL REGIME: Require stronger HTF confirmation
            # LONG: Fisher long + 1w bull + 1d bull
            if fisher_long and htf_1w_bull and htf_1d_bull:
                desired_signal = SIZE * 0.8
            # SHORT: Fisher short + 1w bear + 1d bear
            elif fisher_short and htf_1w_bear and htf_1d_bear:
                desired_signal = -SIZE * 0.8
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
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