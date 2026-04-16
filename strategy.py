#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 12h EMA34 trend filter and 1d volume spike confirmation.
# Long when Bull Power > 0 AND price > 12h EMA34 AND 1d volume > 2.0x 20-period average.
# Short when Bear Power < 0 AND price < 12h EMA34 AND 1d volume > 2.0x 20-period average.
# Exit when price crosses 12h EMA34 in opposite direction.
# Uses discrete position size 0.25. Elder Ray measures bull/bear power relative to EMA,
# 12h EMA34 provides medium-term trend filter, volume confirmation reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data once before loop for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 34:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # === 6h Indicators: EMA34 for Elder Ray ===
    ema34_6h = pd.Series(close_6h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Bull Power = High - EMA34
    bull_power = high_6h - ema34_6h
    # Bear Power = Low - EMA34
    bear_power = low_6h - ema34_6h
    
    # Align 6h indicators to lower timeframe
    ema34_6h_aligned = align_htf_to_ltf(prices, df_6h, ema34_6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Get 12h data once before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA34 trend filter ===
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Volume spike filter ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_6h_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        ema34_6h_val = ema34_6h_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        ema34_12h_val = ema34_12h_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 2.0x 20-period average (using 1d volume MA)
        vol_filter = vol > 2.0 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below 12h EMA34
            if price < ema34_12h_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above 12h EMA34
            if price > ema34_12h_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND price > 12h EMA34 AND volume confirmation
            if bull_power_val > 0 and price > ema34_12h_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power < 0 AND price < 12h EMA34 AND volume confirmation
            elif bear_power_val < 0 and price < ema34_12h_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_12hEMA34_1dVolumeSpike_V1"
timeframe = "6h"
leverage = 1.0