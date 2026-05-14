#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with volume confirmation and 1d EMA trend filter.
# Long when price breaks above daily Camarilla R1 with volume > 1.5x 24-period average and price > 1d EMA(34).
# Short when price breaks below daily Camarilla S1 with volume > 1.5x 24-period average and price < 1d EMA(34).
# Exit when price returns to Camarilla pivot point (PP).
# Uses Camarilla levels from daily timeframe for structure, volume surge for conviction, EMA for trend filter.
# Designed for ~15-30 trades/year per symbol.
name = "12h_Camarilla_R1S1_Volume_EMA34_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # EMA(34) on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 24-period average (24 * 12h = 12 days, using shorter for responsiveness)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_34_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and above EMA
            if close_val > r1_val and vol_filter and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge and below EMA
            elif close_val < s1_val and vol_filter and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point (PP)
            if close_val <= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point (PP)
            if close_val >= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals