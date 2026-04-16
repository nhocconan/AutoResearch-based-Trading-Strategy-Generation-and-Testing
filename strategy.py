#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (R1, S1) with volume spike confirmation.
# Long when price breaks above R1 with volume > 1.5x average volume.
# Short when price breaks below S1 with volume > 1.5x average volume.
# Exit when price returns to the 1d close (mean reversion to daily equilibrium).
# Uses discrete position size 0.25. Camarilla levels provide intraday support/resistance.
# 1d timeframe filter ensures trading with higher timeframe structure to avoid noise.
# 12h timeframe targets 12-37 trades/year to minimize fee drag.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    # Pivot = (H + L + C) / 3
    # R1 = C + ((H - L) * 1.1 / 12)
    # S1 = C - ((H - L) * 1.1 / 12)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12.0)
    s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12.0)
    
    # Align Camarilla levels to primary timeframe (12h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # === Volume Spike: 1.5x average volume (20-period) ===
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20  # volume MA needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        daily_close = close_1d_aligned[i]
        vol_ma = volume_ma[i]
        price = close[i]
        vol = volume[i]
        
        # Volume spike condition
        volume_spike = vol > (1.5 * vol_ma) if vol_ma > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price returns to or below daily close (mean reversion)
            if price <= daily_close:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price returns to or above daily close (mean reversion)
            if price >= daily_close:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike
            if (price > r1) and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S1 with volume spike
            elif (price < s1) and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dCamarilla_R1S1_VolumeSpike_MeanReversion_V1"
timeframe = "12h"
leverage = 1.0