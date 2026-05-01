#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>1.5x 20-bar MA)
# Camarilla pivot levels provide intraday support/resistance. Breakout of R3/S3 indicates strong momentum.
# 1d EMA34 filter ensures alignment with higher-timeframe trend to avoid counter-trend whipsaws.
# Volume spike confirms institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical Camarilla: H1, H2, H3, H4, L1, L2, L3, L4
    # We use R3 = H3 and S3 = L3 for breakout
    # Formula: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    df_1d['h3'] = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    df_1d['l3'] = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_h3 = df_1d['h3'].values
    camarilla_l3 = df_1d['l3'].values
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 (H3) with volume spike and above 1d EMA34
            if curr_high > camarilla_h3_aligned[i] and curr_close > ema_1d_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 (L3) with volume spike and below 1d EMA34
            elif curr_low < camarilla_l3_aligned[i] and curr_close < ema_1d_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below 1d EMA34 or re-entry below R3
            if curr_close < ema_1d_34_aligned[i] or curr_low < camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above 1d EMA34 or re-entry above S3
            if curr_close > ema_1d_34_aligned[i] or curr_high > camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals