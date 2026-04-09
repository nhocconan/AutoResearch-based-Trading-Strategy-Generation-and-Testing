#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_reversion_v1
# Hypothesis: On 12-hour timeframe, price tends to revert to Camarilla pivot levels from the prior day.
# In both bull and bear markets, price often respects these key levels as support/resistance.
# We enter long near S3/S4 and short near R3/R4 with volume confirmation, using 1-day trend filter.
# Uses daily EMA(34) to filter trades in direction of higher timeframe trend.
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag on 12h chart.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once before loop (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if df_1d is None or len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align to 12h timeframe (waits for daily close)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior daily bar (HLC of previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation on 12h: volume > 1.5x 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 1.5
    
    # Initialize signals
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start loop after warmup period
    for i in range(34, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (profit target) or trend turns bearish
            if price >= r3_aligned[i] or ema_34_aligned[i] > price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 (profit target) or trend turns bullish
            if price <= s3_aligned[i] or ema_34_aligned[i] < price:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long near S4/S3 with volume confirmation and bullish 1d trend
            if (price <= s4_aligned[i] * 1.002 or price <= s3_aligned[i] * 1.002) and \
               volume[i] > vol_threshold[i] and \
               ema_34_aligned[i] < price:  # price above daily EMA = bullish bias
                position = 1
                signals[i] = 0.25
            # Enter short near R3/R4 with volume confirmation and bearish 1d trend
            elif (price >= r4_aligned[i] * 0.998 or price >= r3_aligned[i] * 0.998) and \
                 volume[i] > vol_threshold[i] and \
                 ema_34_aligned[i] > price:  # price below daily EMA = bearish bias
                position = -1
                signals[i] = -0.25
    
    return signals