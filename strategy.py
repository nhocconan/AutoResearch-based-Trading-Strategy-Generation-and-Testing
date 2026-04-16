#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA200 trend filter and 1w volume spike.
# Long when Alligator jaws (13-period smoothed median) crosses above teeth (8-period smoothed median) 
# AND price > 1d EMA200 (bullish regime) AND 1w volume > 2.0x 20-period average.
# Short when jaws crosses below teeth AND price < 1d EMA200 (bearish regime) AND 1w volume spike.
# Exit on opposite Alligator cross or when price crosses EMA200 in opposite direction.
# Uses discrete position size 0.25. Designed to catch trends with volume confirmation in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Williams Alligator (Jaws, Teeth, Lips) ===
    median_price = (high + low) / 2.0
    
    # Jaws: 13-period smoothed median, 8 bars ahead
    jaws_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws_raw, 8)
    jaws[:8] = np.nan
    
    # Teeth: 8-period smoothed median, 5 bars ahead
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period smoothed median, 3 bars ahead
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # === 1d EMA200 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 1w Volume Spike (volume > 2.0x 20-period average) ===
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_spike = volume > (2.0 * vol_ma_1w_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 200 periods needed for EMA200)
    warmup = 250
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if jaws crosses below teeth (Alligator sleeping)
            if jaws[i] < teeth[i]:
                exit_signal = True
            # Exit if price crosses below EMA200 (trend change)
            elif price < ema_200_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if jaws crosses above teeth (Alligator waking)
            if jaws[i] > teeth[i]:
                exit_signal = True
            # Exit if price crosses above EMA200 (trend change)
            elif price > ema_200_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Jaws crosses above teeth AND price > EMA200 AND volume spike
            if jaws[i] > teeth[i] and price > ema_200_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Jaws crosses below teeth AND price < EMA200 AND volume spike
            elif jaws[i] < teeth[i] and price < ema_200_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA200_1wVolumeSpike_V1"
timeframe = "12h"
leverage = 1.0