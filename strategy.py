#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume regime filter
# - Long when price breaks above Camarilla R4 (1d) AND 1d volume > 1.5x 20-period average
# - Short when price breaks below Camarilla S4 (1d) AND 1d volume > 1.5x 20-period average
# - Exit when price retouches Camarilla PP (pivot point) from 1d
# - Camarilla levels from higher timeframe provide structure that works in ranging markets
# - Volume regime filter ensures breakouts occur with participation, reducing false signals
# - Target: 12-37 trades/year on 6h (50-150 total over 4 years) to avoid fee drag

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_pp = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_r4 = np.full_like(close_1d, np.nan, dtype=float)
    camarilla_s4 = np.full_like(close_1d, np.nan, dtype=float)
    
    for i in range(1, len(df_1d)):
        # Use previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        camarilla_pp[i] = (prev_high + prev_low + prev_close) / 3
        camarilla_r4[i] = camarilla_pp[i] + (prev_high - prev_low) * 1.1 / 2
        camarilla_s4[i] = camarilla_pp[i] - (prev_high - prev_low) * 1.1 / 2
    
    # Align HTF Camarilla levels to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(vol_1d, np.nan, dtype=float)
    for i in range(19, len(vol_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Align HTF volume average to 6h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume regime condition: current 1d volume > 1.5x 20-period average
        # Need to get current 1d volume - we'll use the aligned 1d volume series
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        vol_regime = not np.isnan(vol_1d_aligned[i]) and not np.isnan(vol_ma_20_aligned[i]) and \
                     vol_1d_aligned[i] > 1.5 * vol_ma_20_aligned[i]
        
        close_now = prices['close'].values[i]
        pp_now = camarilla_pp_aligned[i]
        r4_now = camarilla_r4_aligned[i]
        s4_now = camarilla_s4_aligned[i]
        
        # Camarilla breakout signals
        breakout_up = close_now > r4_now   # price breaks above Camarilla R4
        breakout_down = close_now < s4_now  # price breaks below Camarilla S4
        retouch_pp = abs(close_now - pp_now) < 0.001 * pp_now  # price retouches PP (within 0.1%)
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla R4 AND volume regime
            if breakout_up and vol_regime:
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla S4 AND volume regime
            elif breakout_down and vol_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price retouches Camarilla PP
            exit_long = (position == 1 and retouch_pp)
            exit_short = (position == -1 and retouch_pp)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals