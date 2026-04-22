#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with volume surge and ADX trend filter
    # Uses actual pivot levels (not OHLC-based) from prior day for structure
    # ADX(14) > 25 ensures we trade in trending markets only
    # Volume surge (2x 20-period MA) confirms breakout strength
    # Works in bull/bear: breakouts from key levels with momentum capture moves
    
    # Load daily data for Camarilla pivots (based on prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on prior day)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) * 0.275
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) * 0.275
    
    # Align to 4h - each daily level applies to all 4h bars of that day
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Load ADX data from 4h (same timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Price and volume data
    volume = prices['volume'].values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume spike and ADX > 25 (trending up)
            if close[i] > camarilla_r1_aligned[i] and vol_spike[i] and adx[i] > 25 and plus_di[i] > minus_di[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume spike and ADX > 25 (trending down)
            elif close[i] < camarilla_s1_aligned[i] and vol_spike[i] and adx[i] > 25 and minus_di[i] > plus_di[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to pivot point (close of prior day) or opposite level touch
            # Pivot point = (high + low + close)/3
            pivot_point = (high_1d + low_1d + close_1d) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
            
            if position == 1:
                if close[i] < pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_ADX25_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0