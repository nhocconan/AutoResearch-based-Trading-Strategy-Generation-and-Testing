#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation
# Elder Ray measures bull/bear power: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In trending markets: long when Bull Power > 0 and Bear Power rising (less negative)
#                 short when Bear Power < 0 and Bull Power falling (less positive)
# Uses 1d EMA50 for trend alignment to avoid counter-trend trades
# Volume confirmation (>1.4x 20-period average) ensures participation
# Designed for ~12-30 trades/year on 6h timeframe to minimize fee drag

name = "6h_ElderRay_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Calculate 20-period average volume for confirmation (on 6h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for EMA13 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Bull Power turns negative OR Bear Power rises above -0.5*ATR proxy
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR Bull Power falls below 0.5*ATR proxy
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.4x 20-period average
            vol_confirm = curr_volume > 1.4 * curr_vol_ma
            
            # Long entry: Bull Power > 0 (bulls in control) AND Bear Power rising (less negative)
            # AND price above 1d EMA50 (uptrend alignment)
            if vol_confirm and curr_close > curr_ema50_1d:
                if curr_bull_power > 0 and curr_bear_power > bear_power[i-1]:
                    signals[i] = 0.25
                    position = 1
            # Short entry: Bear Power < 0 (bears in control) AND Bull Power falling (less positive)
            # AND price below 1d EMA50 (downtrend alignment)
            elif vol_confirm and curr_close < curr_ema50_1d:
                if curr_bear_power < 0 and curr_bull_power < bull_power[i-1]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
    
    return signals