#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend + volume spike
# Alligator jaws/teeth/lips (SMAs 13/8/5) define trend: 
#   Bullish: lips > teeth > jaws (green alignment)
#   Bearish: jaws > teeth > lips (red alignment)
# Enter long when bullish alignment + price > 1d EMA34 + volume > 2.0x average
# Enter short when bearish alignment + price < 1d EMA34 + volume > 2.0x average
# Uses discrete sizing (0.25) and tight entry conditions to target 12-37 trades/year.
# Alligator filters choppy markets; 1d EMA34 ensures higher timeframe trend alignment;
# volume confirms breakout strength. Works in bull/bear via trend filter.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: SMAs of median price (hlc3)
    hlc3 = (high + low + close) / 3.0
    # Jaws: SMA(13, 8) - slowest
    jaws = pd.Series(hlc3).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    # Teeth: SMA(8, 5) - middle
    teeth = pd.Series(hlc3).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    # Lips: SMA(5, 3) - fastest
    lips = pd.Series(hlc3).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34 and Alligator (need sufficient lookback)
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaws = jaws[i]
        
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Alligator alignment breaks (lips <= teeth or teeth <= jaws)
            # 2. Price crosses below 1d EMA34 (trend change)
            if (curr_lips <= curr_teeth or
                curr_teeth <= curr_jaws or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Alligator alignment breaks (jaws <= teeth or teeth <= lips)
            # 2. Price crosses above 1d EMA34 (trend change)
            if (curr_jaws <= curr_teeth or
                curr_teeth <= curr_lips or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Bullish alignment: lips > teeth > jaws
            bullish = (curr_lips > curr_teeth) and (curr_teeth > curr_jaws)
            # Bearish alignment: jaws > teeth > lips
            bearish = (curr_jaws > curr_teeth) and (curr_teeth > curr_lips)
            
            # Long entry: bullish alignment + price > 1d EMA34 + volume confirm
            if bullish and (curr_close > curr_ema_34_1d) and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment + price < 1d EMA34 + volume confirm
            elif bearish and (curr_close < curr_ema_34_1d) and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals