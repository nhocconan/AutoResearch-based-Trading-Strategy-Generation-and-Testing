#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R1 (1d) AND 1d close > 1d EMA50 (uptrend) AND 12h volume > 1.8x 20-period average.
# Short when price breaks below S1 (1d) AND 1d close < 1d EMA50 (downtrend) AND 12h volume > 1.8x 20-period average.
# Uses discrete position size 0.25. Camarilla levels provide structure, 1d EMA50 ensures trend alignment,
# volume spike confirms breakout strength. Designed to capture momentum moves in both bull and bear markets.
# Target: 80-120 trades over 4 years (20-30/year) within HARD MAX: 200 total.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Get 1d data once before loop for Camarilla levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Camarilla R1, S1 and EMA50 ===
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_1d = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls back below R1 or volume spike ends
            if price < r1 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises back above S1 or volume spike ends
            if price > s1 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 AND 1d close > 1d EMA50 (uptrend) AND volume spike
            if price > r1 and close_1d[i] > ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below S1 AND 1d close < 1d EMA50 (downtrend) AND volume spike
            elif price < s1 and close_1d[i] < ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_CamarillaR1S1_Breakout_1dEMA50_VolumeSpike_V1"
timeframe = "12h"
leverage = 1.0