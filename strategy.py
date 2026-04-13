#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels with volume spike and ADX trend filter.
# Camarilla levels provide precise support/resistance for reversals.
# Volume spikes confirm participation at key levels.
# ADX ensures trades align with prevailing trend to avoid chop.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.zeros(len(close_1d))  # Resistance level
    camarilla_l4 = np.zeros(len(close_1d))  # Support level
    
    for i in range(1, len(close_1d)):
        range_1d = high_1d[i-1] - low_1d[i-1]
        camarilla_h4[i] = close_1d[i-1] + range_1d * 1.1 / 2
        camarilla_l4[i] = close_1d[i-1] - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate ADX (14-period) for trend strength
    period = 14
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        plus_dm[i] = max(high[i] - high[i-1], 0)
        minus_dm[i] = max(low[i-1] - low[i], 0)
        if plus_dm[i] == minus_dm[i]:
            plus_dm[i] = 0
            minus_dm[i] = 0
        elif plus_dm[i] < minus_dm[i]:
            plus_dm[i] = 0
        else:
            minus_dm[i] = 0
    
    atr = np.zeros(n)
    for i in range(period, n):
        atr[i] = np.mean(tr[i-period+1:i+1])
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period*2, n):
        plus_dm_ma = np.mean(plus_dm[i-period+1:i+1])
        minus_dm_ma = np.mean(minus_dm[i-period+1:i+1])
        atr_ma = atr[i]
        if atr_ma > 0:
            plus_di[i] = 100 * plus_dm_ma / atr_ma
            minus_di[i] = 100 * minus_dm_ma / atr_ma
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    for i in range(period*3, n):
        adx[i] = np.mean(dx[i-period+1:i+1])
    
    # Calculate average volume (20-period) for volume spike
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        camarilla_h4_val = camarilla_h4_aligned[i]
        camarilla_l4_val = camarilla_l4_aligned[i]
        adx_val = adx[i]
        
        # Volume spike: current volume > 2x average volume
        volume_spike = vol > 2.0 * avg_vol
        
        # ADX filter: trend strength > 25
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long: price touches Camarilla L4 support + volume spike + uptrend
            if (abs(price - camarilla_l4_val) < 0.001 * camarilla_l4_val and  # Within 0.1%
                volume_spike and
                trend_filter and
                price > camarilla_h4_val):  # Price above resistance for confirmation
                position = 1
                signals[i] = position_size
            # Short: price touches Camarilla H4 resistance + volume spike + downtrend
            elif (abs(price - camarilla_h4_val) < 0.001 * camarilla_h4_val and  # Within 0.1%
                  volume_spike and
                  trend_filter and
                  price < camarilla_l4_val):  # Price below support for confirmation
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches Camarilla H4 resistance or volume drops
            if (price >= camarilla_h4_val or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches Camarilla L4 support or volume drops
            if (price <= camarilla_l4_val or 
                vol < 0.5 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Volume_Spike_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0