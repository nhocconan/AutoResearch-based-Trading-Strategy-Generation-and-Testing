#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 AND Bear Power rising AND 12h EMA(21) rising AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power falling AND 12h EMA(21) falling AND volume > 1.5x average
# Exit when Bull Power < 0 (for long) or Bear Power > 0 (for short) OR opposite signal
# Uses 13-period EMA for Elder Ray calculation, 12h EMA for trend filter, volume for confirmation
# Designed to capture institutional buying/selling pressure with trend alignment
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 13-period EMA for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    
    # Calculate Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate EMA on 12h (21-period) for trend filter
    ema_21_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get EMA values aligned to 6h timeframe
        ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h.values)
        ema_val = ema_21_aligned[i]
        ema_prev = ema_21_aligned[i-1]
        
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        bull_prev = bull_power[i-1]
        bear_prev = bear_power[i-1]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Bull Power > 0 AND Bear Power rising AND 12h EMA rising AND volume confirmation
            if (bull_val > 0 and bear_val > bear_prev and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Bear Power < 0 AND Bull Power falling AND 12h EMA falling AND volume confirmation
            elif (bear_val < 0 and bull_val < bull_prev and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power < 0 OR opposite signal
            if (bull_val < 0 or 
                (bear_val < 0 and bull_val < bull_prev and ema_val < ema_prev and vol > vol_threshold)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power > 0 OR opposite signal
            if (bear_val > 0 or 
                (bull_val > 0 and bear_val > bear_prev and ema_val > ema_prev and vol > vol_threshold)):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_12hEMA_Volume"
timeframe = "6h"
leverage = 1.0