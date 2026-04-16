#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Elder Ray (Bull/Bear Power) + 1d volume filter.
# Long when Williams %R(14) < -80 (oversold) AND 1d Bull Power > 0 AND volume > 1.5x 20-period average.
# Short when Williams %R(14) > -20 (overbought) AND 1d Bear Power < 0 AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position size 0.25. Williams %R identifies extremes, Elder Ray confirms bull/bear momentum via EMA(13),
# volume filter ensures participation. Target: 60-150 total trades over 4 years (15-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Elder Ray and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Elder Ray (Bull/Bear Power) and volume MA ===
    # EMA(13) of close
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13
    # Volume MA(20)
    vol_ma_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Get 6h data for Williams %R
    # Williams %R(14) = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, -50)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(williams_r[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_ma_val = vol_ma_20_aligned[i]
        wr = williams_r[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol > 1.5 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 (momentum fading)
            if wr > -50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 (momentum fading)
            if wr < -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: oversold + bull power positive + volume confirmation
            if wr < -80 and bull_val > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: overbought + bear power negative + volume confirmation
            elif wr > -20 and bear_val < 0 and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR_1dElderRay_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0