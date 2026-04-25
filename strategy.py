#!/usr/bin/env python3
"""
12h Williams Alligator with 1w EMA34 Trend Filter and Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trending vs ranging markets. 
In strong trends (alligator "awake"), we trade breakouts in the direction of the 1w EMA34 trend.
Volume spikes confirm momentum. Designed for low trade frequency (12-37/year) to minimize fee drag.
Works in both bull (long breakouts) and bear (short breakouts) by using 1w HTF trend as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter (weekly trend)
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams Alligator on primary timeframe (12h)
    # Jaw: 13-period SMMA smoothed 8 periods ahead
    # Teeth: 8-period SMMA smoothed 5 periods ahead  
    # Lips: 5-period SMMA smoothed 3 periods ahead
    def smma(series, period):
        """Smoothed Moving Average"""
        if len(series) < period:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        sma = np.mean(series[:period])
        result[period-1] = sma
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(high, 13)  # Using high for jaw (blue line)
    teeth = smma(high, 8)  # Using high for teeth (red line)
    lips = smma(high, 5)   # Using high for lips (green line)
    
    # Shift to avoid look-ahead (Alligator uses future smoothing)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator and EMA
    start_idx = max(50, 34)  # Alligator warmup, EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator signals: 
        # Bullish: Lips > Teeth > Jaw (green > red > blue) - Mouth opening up
        # Bearish: Lips < Teeth < Jaw (green < red < blue) - Mouth opening down
        bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Trend filter: price relative to 1w EMA34
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + trend + volume
            # Long: Bullish Alligator AND bullish bias AND volume spike
            long_entry = bullish_alligator and bullish_bias and vol_spike
            # Short: Bearish Alligator AND bearish bias AND volume spike
            short_entry = bearish_alligator and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator turns bearish OR loss of bullish bias
            if bearish_alligator or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator turns bullish OR loss of bearish bias
            if bullish_alligator or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0