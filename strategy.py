#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and 1d EMA trend filter
# Long at S1 support when price > 1d EMA34 + volume spike
# Short at R1 resistance when price < 1d EMA34 + volume spike
# Exit at midpoint (M) or when trend reverses
# Camarilla levels work well in ranging markets, EMA filter avoids counter-trend trades
# Volume spike ensures institutional participation
# Target: 20-40 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # HLC = (High + Low + Close) / 3
    hlc = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    S1 = hlc - (range_hl * 1.1 / 12)
    S2 = hlc - (range_hl * 1.1 / 6)
    S3 = hlc - (range_hl * 1.1 / 4)
    R1 = hlc + (range_hl * 1.1 / 12)
    R2 = hlc + (range_hl * 1.1 / 6)
    R3 = hlc + (range_hl * 1.1 / 4)
    M = hlc  # Midpoint
    
    # Use previous day's levels
    S1_prev = np.roll(S1, 1)
    R1_prev = np.roll(R1, 1)
    M_prev = np.roll(M, 1)
    S1_prev[0] = np.nan
    R1_prev[0] = np.nan
    M_prev[0] = np.nan
    
    # Align to 4h
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1_prev)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1_prev)
    M_aligned = align_htf_to_ltf(prices, df_1d, M_prev)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(M_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        s1 = S1_aligned[i]
        r1 = R1_aligned[i]
        m = M_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long at S1 support with uptrend and volume spike
            if price <= s1 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short at R1 resistance with downtrend and volume spike
            elif price >= r1 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit at midpoint or if trend turns down
                if price >= m or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit at midpoint or if trend turns up
                if price <= m or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_S1R1_EMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0