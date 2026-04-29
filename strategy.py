#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Bull Power = High - EMA(34), Bear Power = Low - EMA(34)
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d EMA34 (uptrend) with volume spike
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d EMA34 (downtrend) with volume spike
# Volume confirmation ensures institutional participation. Works in both bull (buy strength) and bear (sell weakness).
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA34 to 6h timeframe (completed 1d bar only)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h EMA(34) for Elder Ray
    ema_34_6h = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Elder Ray components
    bull_power = high - ema_34_6h  # Bull Power: High - EMA(34)
    bear_power = low - ema_34_6h   # Bear Power: Low - EMA(34)
    
    # Slope of Bear Power (rising = less negative) and Bull Power (falling = less positive)
    # Using 3-period change for smoothing
    bear_power_slope = bear_power - np.roll(bear_power, 3)
    bull_power_slope = bull_power - np.roll(bull_power, 3)
    # First 3 values will be invalid (set to 0)
    bear_power_slope[:3] = 0
    bull_power_slope[:3] = 0
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 34)  # warmup for EMA and Elder Ray
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_bull_slope = bull_power_slope[i]
        curr_bear_slope = bear_power_slope[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power positive AND Bear Power rising (less negative) AND uptrend with volume
            if (curr_bull_power > 0 and 
                curr_bear_slope > 0 and 
                curr_close > curr_ema_34 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND Bull Power falling (less positive) AND downtrend with volume
            elif (curr_bear_power < 0 and 
                  curr_bull_slope < 0 and 
                  curr_close < curr_ema_34 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit when Bull Power fades OR Bear Power turns negative
            if curr_bull_power <= 0 or curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when Bear Power fades OR Bull Power turns positive
            if curr_bear_power >= 0 or curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals