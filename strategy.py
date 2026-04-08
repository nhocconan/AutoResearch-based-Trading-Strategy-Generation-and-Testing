#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Hypothesis: 12-hour Camarilla pivot reversal with volume confirmation and 1-day ATR regime filter.
# Long: price <= Camarilla L3 AND volume > 1.3x 20-period average volume AND daily ATR < 0.03 * price (low volatility regime).
# Short: price >= Camarilla H3 AND volume > 1.3x 20-period average volume AND daily ATR < 0.03 * price.
# Exit: price crosses Camarilla midpoint (H3/L3 average) or opposite H3/L3 touch with volume.
# Designed to capture mean-reversion bounces at institutional pivot levels during low-volatility periods,
# which works in both bull (buy dips) and bear (sell rallies) markets with strict entry criteria.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12-hour Camarilla levels from previous bar's OHLC
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_mid = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's OHLC to calculate current bar's Camarilla levels
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        camarilla_h3[i] = prev_close + range_val * 1.1 / 4
        camarilla_l3[i] = prev_close - range_val * 1.1 / 4
        camarilla_mid[i] = (camarilla_h3[i] + camarilla_l3[i]) / 2
    
    # 20-period average volume
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1-day ATR(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])
    
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        atr_1d[i] = np.mean(tr[i-14:i])
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        mid = camarilla_mid[i]
        atr_1d_val = atr_1d_aligned[i]
        
        if np.isnan(h3) or np.isnan(l3) or np.isnan(mid) or np.isnan(avg_vol) or np.isnan(atr_1d_val):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.3 * avg_vol
        low_vol_regime = atr_1d_val < 0.03 * price  # ATR < 3% of price
        
        if position == 1:  # Long position
            if price > mid or (price >= h3 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price < mid or (price <= l3 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price <= l3 and vol_surge and low_vol_regime:
                position = 1
                signals[i] = 0.25
            elif price >= h3 and vol_surge and low_vol_regime:
                position = -1
                signals[i] = -0.25
    
    return signals