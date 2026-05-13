# The Real 4H Supertrend Volatility Breakout
# The Real 4H Supertrend Volatility Breakout
#!/usr/bin/env python3
"""
4H_Supertrend_Volatility_Breakout
Hypothesis: Supertrend identifies trend direction, Bollinger Bands volatility expansion confirms breakout strength, and volume surge validates institutional participation. Works in bull markets by catching strong uptrends and in bear markets by catching sharp reversals or counter-trend bounces with volume confirmation.
"""

name = "4H_Supertrend_Volatility_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Supertrend
    atr_period = 10
    atr_multiplier = 3.0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = np.zeros_like(close)
    atr[:atr_period] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    hl2 = (high + low) / 2
    upper_band = hl2 + (atr_multiplier * atr)
    lower_band = hl2 - (atr_multiplier * atr)
    
    supertrend = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    for i in range(1, len(close)):
        if close[i] > upper_band[i-1]:
            supertrend[i] = 1
        elif close[i] < lower_band[i-1]:
            supertrend[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            if supertrend[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if supertrend[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
    
    # Get 1d data for Bollinger Bands volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Bollinger Bands on daily
    bb_period = 20
    bb_std_dev = 2.0
    
    sma_20 = np.zeros_like(close_1d)
    for i in range(bb_period-1, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-bb_period+1:i+1])
    
    bb_std = np.zeros_like(close_1d)
    for i in range(bb_period-1, len(close_1d)):
        bb_std[i] = np.std(close_1d[i-bb_period+1:i+1])
    
    upper_bb = sma_20 + (bb_std_dev * bb_std)
    lower_bb = sma_20 - (bb_std_dev * bb_std)
    bb_width = upper_bb - lower_bb
    
    # Calculate Bollinger Band width percentile (volatility regime)
    bb_width_percentile = np.zeros_like(bb_width)
    lookback = 50
    for i in range(lookback, len(bb_width)):
        if np.all(~np.isnan(bb_width[i-lookback:i+1])):
            bb_width_percentile[i] = np.percentile(bb_width[i-lookback:i+1], 50)  # Median
        else:
            bb_width_percentile[i] = np.nan
    
    # Align 1d indicators to 4h timeframe
    supertrend_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            supertrend_1d[i] = 1 if close_1d[i] > (high_1d[i] + low_1d[i])/2 else -1
        else:
            prev_close = close_1d[i-1]
            # Simplified 1d Supertrend for trend filter
            hl2_1d = (high_1d[i] + low_1d[i]) / 2
            # Use previous day's ATR approximation
            if i >= atr_period:
                atr_1d = np.mean(np.maximum(high_1d[i-atr_period:i+1] - low_1d[i-atr_period:i+1],
                                          np.maximum(np.abs(high_1d[i-atr_period:i+1] - np.roll(close_1d, 1)[i-atr_period:i+1]),
                                                   np.abs(low_1d[i-atr_period:i+1] - np.roll(close_1d, 1)[i-atr_period:i+1]))))
                upper_1d = hl2_1d + (atr_multiplier * atr_1d)
                lower_1d = hl2_1d - (atr_multiplier * atr_1d)
                
                if prev_close > upper_1d:
                    supertrend_1d[i] = 1
                elif prev_close < lower_1d:
                    supertrend_1d[i] = -1
                else:
                    supertrend_1d[i] = supertrend_1d[i-1]
            else:
                supertrend_1d[i] = supertrend_1d[i-1] if i > 0 else 1
    
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend[i]) or np.isnan(supertrend_1d_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility expansion condition: BB width above median (expanding volatility)
        vol_expansion = bb_width_percentile_aligned[i] > 0  # Above median
        
        # Volume spike condition: current volume > 1.8x 20-period average
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Supertrend uptrend + volatility expansion + volume spike + 1d uptrend
            if (supertrend[i] == 1 and vol_expansion and vol_spike and 
                supertrend_1d_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # SHORT: Supertrend downtrend + volatility expansion + volume spike + 1d downtrend
            elif (supertrend[i] == -1 and vol_expansion and vol_spike and 
                  supertrend_1d_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Supertrend reversal or loss of volatility/volume
            if (supertrend[i] == -1 or not vol_expansion or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Supertrend reversal or loss of volatility/volume
            if (supertrend[i] == 1 or not vol_expansion or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals