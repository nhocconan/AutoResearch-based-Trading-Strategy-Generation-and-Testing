#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Long when Bull Power > 0 AND price > 12h EMA(50) AND volume > 1.5x 20-period average.
# Short when Bear Power > 0 AND price < 12h EMA(50) AND volume > 1.5x 20-period average.
# Exit when Bull/Bear Power crosses zero (momentum exhaustion).
# Uses discrete position size 0.25. Elder Ray measures bull/bear strength relative to EMA.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data once before loop for EMA calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    
    # === 6h Indicators: EMA(50) for trend filter ===
    ema_50_6h = pd.Series(close_6h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_50_6h)
    
    # === 6h Indicators: EMA(13) for Elder Ray calculation ===
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_13_6h)
    
    # Get 12h data once before loop for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # === 12h Indicators: Volume spike filter ===
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_6h_aligned[i]) or np.isnan(ema_13_6h_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        ema_50_val = ema_50_6h_aligned[i]
        ema_13_val = ema_13_6h_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Elder Ray components
        bull_power = high[i] - ema_13_val
        bear_power = ema_13_val - low[i]
        
        # Volume filter: volume > 1.5x 20-period average (using 12h volume MA)
        vol_filter = vol > 1.5 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power crosses zero (momentum exhaustion)
            if bull_power <= 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power crosses zero (momentum exhaustion)
            if bear_power <= 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND price > 6h EMA(50) AND volume confirmation
            if bull_power > 0 and price > ema_50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power > 0 AND price < 6h EMA(50) AND volume confirmation
            elif bear_power > 0 and price < ema_50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_EMA50_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0