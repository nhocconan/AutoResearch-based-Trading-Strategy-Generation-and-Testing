#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour price action at daily VWAP with volume confirmation and ATR-based trend filter.
# Long when: Price crosses above daily VWAP, volume > 1.5x 20-period average, ATR(14) rising
# Short when: Price crosses below daily VWAP, volume > 1.5x 20-period average, ATR(14) rising
# Exit when: Price crosses back through daily VWAP
# VWAP acts as dynamic support/resistance, volume confirms institutional interest, ATR rising filters for trending moves.
# Works in bull (buy VWAP bounces) and bear (sell VWAP rejections).
name = "4h_VWAP_Cross_Volume_ATRTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP: typical price * volume / cumulative volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_num = np.cumsum(typical_price_1d * volume_1d)
    vwap_den = np.cumsum(volume_1d)
    vwap_1d = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price_1d)
    
    # Calculate ATR(14) for trend filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1D VWAP to 4H timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Wait for volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_now = atr[i]
        atr_prev = atr[i-1] if i > 0 else atr_now
        
        if position == 0:
            # Long entry: Price crosses above VWAP, volume spike, ATR rising
            if (price > vwap and close[i-1] <= vwap and 
                vol > 1.5 * vol_ma and atr_now > atr_prev):
                signals[i] = 0.25
                position = 1
            # Short entry: Price crosses below VWAP, volume spike, ATR rising
            elif (price < vwap and close[i-1] >= vwap and 
                  vol > 1.5 * vol_ma and atr_now > atr_prev):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below VWAP
            if price < vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above VWAP
            if price > vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals