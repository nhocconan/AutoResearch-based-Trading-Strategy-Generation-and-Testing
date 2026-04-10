#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1d ATR filter
# - Long when price breaks above 12h Camarilla R4 level AND 1d volume > 1.2x 20-period volume SMA AND 12h ATR(14) > 12h ATR(50) * 0.8
# - Short when price breaks below 12h Camarilla S4 level AND 1d volume > 1.2x 20-period volume SMA AND 12h ATR(14) > 12h ATR(50) * 0.8
# - Exit: price retreats to Camarilla pivot point (PP) OR volatility drops (ATR ratio < 0.6)
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 12h timeframe to stay within fee drag limits
# - Uses Camarilla levels from 1d timeframe for structure, 12h for execution timing
# - ATR filter ensures we only trade during sufficient volatility, reducing whipsaw in ranging markets

name = "12h_1d_camarilla_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_range = high_1d - low_1d
    camarilla_r4 = camarilla_pp + camarilla_range * 1.1 / 2.0
    camarilla_s4 = camarilla_pp - camarilla_range * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 12h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar's TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / np.where(atr_50 == 0, 1, atr_50)  # Avoid division by zero
    
    for i in range(60, n):  # Start after warmup for ATR and other indicators
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.2x 20-period volume SMA AND 1d volume > 1.2x 20-period volume SMA
        vol_sma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm_12h = volume[i] > 1.2 * vol_sma_20_12h[i] if not np.isnan(vol_sma_20_12h[i]) else False
        vol_confirm_1d = volume_1d[i] > 1.2 * volume_sma_20_1d_aligned[i] if i < len(volume_1d) and not np.isnan(volume_sma_20_1d_aligned[i]) else False
        vol_confirm = vol_confirm_12h and vol_confirm_1d
        
        # ATR volatility filter: current ATR ratio > 0.8 (ensures sufficient volatility)
        vol_filter = atr_ratio[i] > 0.8
        
        # Camarilla breakout signals (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > camarilla_r4_aligned[i-1]
        breakout_down = close[i] < camarilla_s4_aligned[i-1]
        
        # Exit conditions: price retreats to pivot point OR volatility drops significantly
        exit_long = close[i] < camarilla_pp_aligned[i] or atr_ratio[i] < 0.6
        exit_short = close[i] > camarilla_pp_aligned[i] or atr_ratio[i] < 0.6
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and vol_filter:
                position = 1
                signals[i] = 0.25
            elif breakout_down and vol_confirm and vol_filter:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals