#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels + 1d volume confirmation
# Long when price breaks above weekly R4 with 1d volume > 1.5x 20-period average
# Short when price breaks below weekly S4 with 1d volume > 1.5x 20-period average
# Exit when price returns to weekly Pivot Point (PP)
# Uses discrete position sizing 0.25 to target ~12-25 trades/year
# Weekly Camarilla levels provide structure; volume confirms institutional interest
# Works in bull/bear markets: breakouts capture strong moves, PP exit limits losses in reversals

name = "6h_1w_1d_camarilla_breakout_v1"
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
    open_time = prices['open_time'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Load daily data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1/2
    # S4 = PP - (H - L) * 1.1/2
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    r4_1w = pp_1w + (high_1w - low_1w) * 1.1 / 2.0
    s4_1w = pp_1w - (high_1w - low_1w) * 1.1 / 2.0
    
    # Calculate 1d average volume (20-period)
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly Camarilla levels to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Align daily average volume to 6h timeframe
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or
            np.isnan(s4_1w_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled)
        # Scale 1d avg volume to 6h equivalent (approximate: 1d has ~4 bars of 6h)
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit long when price returns to or below weekly PP
            if close[i] <= pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when price returns to or above weekly PP
            if close[i] >= pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on weekly R4/S4 breakout with volume confirmation
            if close[i] > r4_1w_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            elif close[i] < s4_1w_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals