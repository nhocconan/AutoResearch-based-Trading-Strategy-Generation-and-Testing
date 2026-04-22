#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and ADX trend filter
# Long when price breaks above H4 with ADX > 25 (trending) + volume spike
# Short when price breaks below L4 with ADX > 25 (trending) + volume spike
# Exit when price crosses P (pivot) level or ADX < 20 (range)
# Camarilla levels derived from prior day's range; effective in trending markets with volume confirmation
# Target: 20-40 trades/year to minimize fee drag while capturing breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # S4 = C - ((H-L) * 1.1/2), S3 = C - ((H-L) * 1.1/4), etc.
    # H4 = R3, L4 = S3
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + (range_1d * 1.1 / 4)
    camarilla_l4 = close_1d - (range_1d * 1.1 / 4)
    camarilla_p = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 4h
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    
    # Calculate ADX on 1d for trend filter
    # ADX calculation: +DM, -DM, TR, then smoothed
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(high_1d, 1)), 
                               np.abs(low_1d - np.roll(low_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # first TR
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 24-period average volume for volume spike (1d = 24*4h bars)
    volume = prices['volume'].values
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or 
            np.isnan(p_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        h4 = h4_aligned[i]
        l4 = l4_aligned[i]
        p = p_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 2.5 * 24-period average
        vol_spike = vol > 2.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above H4 + trending + volume spike
            if price > h4 and adx_val > 25.0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below L4 + trending + volume spike
            elif price < l4 and adx_val > 25.0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses P or ADX drops (range)
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below P or trend weakens
                if price < p or adx_val < 20.0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above P or trend weakens
                if price > p or adx_val < 20.0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_H4L4_1dADX_Volume"
timeframe = "4h"
leverage = 1.0