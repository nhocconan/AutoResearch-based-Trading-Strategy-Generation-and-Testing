#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for trend direction and 1d Williams %R for mean-reversion entries
# In trending markets (Supertrend long): buy pullbacks to VWAP with Williams %R oversold
# In ranging markets (Supertrend flat): fade extremes at Williams %R overbought/oversold
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Designed to work in both bull (trend following) and bear (mean reversion in ranges) markets

name = "6h_12h_1d_supertrend_williamsr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(10) for Supertrend
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr_12h, 10)
    
    # Calculate 12h Supertrend
    hl2_12h = (high_12h + low_12h) / 2
    upperband_12h = hl2_12h + 3.0 * atr_12h
    lowerband_12h = hl2_12h - 3.0 * atr_12h
    
    supertrend_12h = np.full(len(close_12h), np.nan)
    direction_12h = np.full(len(close_12h), np.nan)  # 1=uptrend, -1=downtrend
    
    for i in range(len(close_12h)):
        if np.isnan(atr_12h[i]):
            continue
        if i == 0:
            supertrend_12h[i] = hl2_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h[i-1] > supertrend_12h[i-1]:
                upperband_12h[i] = min(upperband_12h[i], upperband_12h[i-1])
            else:
                lowerband_12h[i] = max(lowerband_12h[i], lowerband_12h[i-1])
            
            if close_12h[i] > upperband_12h[i]:
                direction_12h[i] = 1
            elif close_12h[i] < lowerband_12h[i]:
                direction_12h[i] = -1
            else:
                direction_12h[i] = direction_12h[i-1]
            
            if direction_12h[i] == 1:
                supertrend_12h[i] = lowerband_12h[i]
            else:
                supertrend_12h[i] = upperband_12h[i]
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williamsr_1d = -100 * (hh_1d - close_1d) / (hh_1d - ll_1d)
    williamsr_1d = np.where((hh_1d - ll_1d) == 0, -50, williamsr_1d)
    
    # Calculate 6h VWAP (volume-weighted average price)
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Align indicators to 6h timeframe
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    williamsr_1d_aligned = align_htf_to_ltf(prices, df_1d, williamsr_1d)
    vwap_aligned = vwap  # already 6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend_12h_aligned[i]) or np.isnan(direction_12h_aligned[i]) or
            np.isnan(williamsr_1d_aligned[i]) or np.isnan(vwap_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if direction_12h_aligned[i] == 1:  # Uptrend
                # Exit long if trend turns down or price breaks below VWAP
                if direction_12h_aligned[i] == -1 or close[i] < vwap_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Downtrend or ranging
                # Exit long if overbought
                if williamsr_1d_aligned[i] > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short position
            if direction_12h_aligned[i] == -1:  # Downtrend
                # Exit short if trend turns up or price breaks above VWAP
                if direction_12h_aligned[i] == 1 or close[i] > vwap_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Uptrend or ranging
                # Exit short if oversold
                if williamsr_1d_aligned[i] < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        
        else:  # Flat
            if direction_12h_aligned[i] == 1:  # Uptrend
                # Enter long on pullback to VWAP with oversold Williams %R
                if close[i] <= vwap_aligned[i] and williamsr_1d_aligned[i] < -80:
                    position = 1
                    signals[i] = 0.25
            elif direction_12h_aligned[i] == -1:  # Downtrend
                # Enter short on bounce to VWAP with overbought Williams %R
                if close[i] >= vwap_aligned[i] and williamsr_1d_aligned[i] > -20:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging (flat trend)
                # Mean reversion: buy oversold, sell overbought
                if williamsr_1d_aligned[i] < -80:
                    position = 1
                    signals[i] = 0.25
                elif williamsr_1d_aligned[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals