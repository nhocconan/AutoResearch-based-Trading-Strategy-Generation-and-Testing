#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h Trend Filter + Volume Spike
# Elder Ray measures bull/bear power by comparing price to EMA(13).
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and Bear Power < 0 with volume spike and 12h EMA34 uptrend.
# Short when Bear Power < 0 and Bull Power < 0 with volume spike and 12h EMA34 downtrend.
# Designed for low trade frequency (12-37/year) with clear trend and momentum conditions.
# Works in bull markets (buy strength) and bear markets (sell weakness).
name = "6h_ElderRay_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA34 slope (using 3-period change)
        if i >= 3:
            ema34_slope = ema34_12h_aligned[i] - ema34_12h_aligned[i-3]
            uptrend = ema34_slope > 0
            downtrend = ema34_slope < 0
        else:
            uptrend = False
            downtrend = False
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, volume spike, 12h uptrend
            long_condition = (bull_power[i] > 0) and (bear_power[i] < 0) and volume_spike[i] and uptrend
            # Short: Bear Power < 0, Bull Power < 0, volume spike, 12h downtrend
            short_condition = (bear_power[i] < 0) and (bull_power[i] < 0) and volume_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bear Power becomes positive (loss of bearish pressure) OR volume dries up
            exit_condition = (bear_power[i] >= 0) or (volume[i] < vol_ma_20[i] * 0.5)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power becomes positive (loss of bullish pressure) OR volume dries up
            exit_condition = (bull_power[i] >= 0) or (volume[i] < vol_ma_20[i] * 0.5)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals