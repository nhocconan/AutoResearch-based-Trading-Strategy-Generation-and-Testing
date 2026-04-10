#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with 1d volume spike and 1w ADX trend filter
# - Long when price touches Camarilla L3 level with volume spike and weekly uptrend (ADX>25)
# - Short when price touches Camarilla H3 level with volume spike and weekly downtrend (ADX>25)
# - Uses 12h timeframe to target 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# - Weekly ADX > 25 ensures we trade with strong weekly trend direction (avoid chop)
# - Volume confirmation: current 12h volume > 2.0x 20-period average to filter weak touches
# - Discrete position sizing (0.25) to minimize fee churn
# - Exit on opposite Camarilla level touch (L3 for shorts, H3 for longs) or close beyond H4/L4

name = "12h_1d_1w_camarilla_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 1d Camarilla levels (from previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_h4 = close_1d + 1.1/2 * (high_1d - low_1d)  # H4
    camarilla_h3 = close_1d + 1.1/4 * (high_1d - low_1d)  # H3
    camarilla_l3 = close_1d - 1.1/4 * (high_1d - low_1d)  # L3
    camarilla_l4 = close_1d - 1.1/2 * (high_1d - low_1d)  # L4
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches L3 (profit target) or closes beyond L4 (stop/reversal)
            if (prices['close'].iloc[i] <= camarilla_l3_aligned[i] or 
                prices['close'].iloc[i] < camarilla_l4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches H3 (profit target) or closes beyond H4 (stop/reversal)
            if (prices['close'].iloc[i] >= camarilla_h3_aligned[i] or 
                prices['close'].iloc[i] > camarilla_h4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla touch with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Long signal: price touches L3 in weekly uptrend
                if abs(prices['close'].iloc[i] - camarilla_l3_aligned[i]) < 0.001 * camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short signal: price touches H3 in weekly downtrend
                elif abs(prices['close'].iloc[i] - camarilla_h3_aligned[i]) < 0.001 * camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals