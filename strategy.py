#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 12h trend filter and volume confirmation
# Long when: Alligator jaws (13-period SMA) > teeth (8-period SMA) > lips (5-period SMA) AND price > 12h EMA34 AND volume > 1.5x 20-period average volume
# Short when: Alligator jaws < teeth < lips AND price < 12h EMA34 AND volume > 1.5x 20-period average volume
# Williams Alligator identifies trend alignment, 12h EMA34 filters counter-trend trades, volume confirmation adds conviction
# Designed for moderate trade frequency (target: 50-150 total trades over 4 years) to balance opportunity and cost on 6h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h EMA34 (trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 6h Williams Alligator ===
    # Jaws: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward  
    # Lips: 5-period SMA, shifted 3 bars forward
    sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    
    jaws = np.roll(sma_13, 8)  # shift forward 8 bars
    teeth = np.roll(sma_8, 5)  # shift forward 5 bars
    lips = np.roll(sma_5, 3)   # shift forward 3 bars
    
    # Fill NaN from rolling with forward values
    for i in range(5):
        if np.isnan(sma_5[i]):
            sma_5[i] = close[i]
    for i in range(8):
        if np.isnan(sma_8[i]):
            sma_8[i] = close[i]
    for i in range(13):
        if np.isnan(sma_13[i]):
            sma_13[i] = close[i]
            
    # === 6h Volume Spike Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup - need enough for Alligator calculation
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(jaws[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_34_val = ema_34_12h_aligned[i]
        jaw_val = jaws[i]
        tooth_val = teeth[i]
        lip_val = lips[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume for spike
        
        # Alligator alignment conditions
        bullish_alignment = jaw_val > tooth_val and tooth_val > lip_val
        bearish_alignment = jaw_val < tooth_val and tooth_val < lip_val
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: bullish Alligator alignment AND price > 12h EMA34 AND volume spike
            if bullish_alignment and price > ema_34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: bearish Alligator alignment AND price < 12h EMA34 AND volume spike
            elif bearish_alignment and price < ema_34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # === EXIT LOGIC ===
        elif position == 1:  # Long position
            # Exit when Alligator alignment turns bearish or price breaks below 12h EMA34
            if not bullish_alignment or price < ema_34_val:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Alligator alignment turns bullish or price breaks above 12h EMA34
            if not bearish_alignment or price > ema_34_val:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_12hEMA34_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0