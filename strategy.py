#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 Trend Filter and Volume Spike
# Uses Elder Ray indicator: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Strong bullish signal when Bull Power > 0 and rising + price above 1d EMA34 + volume spike
# Strong bearish signal when Bear Power < 0 and falling + price below 1d EMA34 + volume spike
# Works in both bull and bear markets by aligning with higher timeframe trend (1d EMA34)
# Volume confirmation filters out weak breakouts
# Target: 12-37 trades/year (50-150 total over 4 years)

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA34 to 6h timeframe (completed 1d bar only)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate rising/falling Elder Ray (1-period change)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_falling = bear_power < np.roll(bear_power, 1)
    # Handle first bar
    bull_power_rising[0] = False
    bear_power_falling[0] = False
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20, 13)  # warmup for EMA34, volume MA, and EMA13
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_bull_rising = bull_power_rising[i]
        curr_bear_falling = bear_power_falling[i]
        curr_ema34 = ema_34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Determine daily trend: price above/below EMA34
        uptrend = curr_close > curr_ema34
        downtrend = curr_close < curr_ema34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in direction of daily trend
            if curr_volume_confirm:
                # Bullish entry: Bull Power > 0 and rising + uptrend
                if curr_bull_power > 0 and curr_bull_rising and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power < 0 and falling + downtrend
                elif curr_bear_power < 0 and curr_bear_falling and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Bull Power <= 0 OR price breaks below EMA34 with volume
            if curr_bull_power <= 0 or (curr_close < curr_ema34 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Bear Power >= 0 OR price breaks above EMA34 with volume
            if curr_bear_power >= 0 or (curr_close > curr_ema34 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals