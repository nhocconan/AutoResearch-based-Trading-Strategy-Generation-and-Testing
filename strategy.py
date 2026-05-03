#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA(34) trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) + price > lips + volume spike + price > 1d EMA(34)
# Short when Alligator jaws > teeth > lips (bearish alignment) + price < lips + volume spike + price < 1d EMA(34)
# Uses 6h Alligator for trend identification and 1d EMA(34) for intermediate trend filter
# Volume spike (2.0x 20-period average) confirms institutional participation
# Designed for low trade frequency (12-37/year on 6h) to minimize fee drag
# Works in both bull (trend continuation) and bear (trend exhaustion) markets

name = "6h_WilliamsAlligator_Volume_1dEMA34_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h
    # Jaws: Blue line - SMMA(13, 8)
    # Teeth: Red line - SMMA(8, 5)
    # Lips: Green line - SMMA(5, 3)
    
    def smma(values, period, shift):
        """Smoothed Moving Average (SMMA) with shift"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(values, np.nan)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(values)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + values[i]) / period
        return np.roll(smma_vals, shift)
    
    jaws = smma(high, 13, 8)   # Blue line
    teeth = smma(low, 8, 5)    # Red line
    lips = smma(close, 5, 3)   # Green line
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(13 for jaws, 8 for teeth, 5 for lips, 20 for volume MA, 34 for 1d EMA)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish Alligator alignment (jaws < teeth < lips) + price > lips + volume spike + price > 1d EMA(34)
            if (jaws[i] < teeth[i] and teeth[i] < lips[i] and 
                close[i] > lips[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator alignment (jaws > teeth > lips) + price < lips + volume spike + price < 1d EMA(34)
            elif (jaws[i] > teeth[i] and teeth[i] > lips[i] and 
                  close[i] < lips[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish Alligator alignment OR price < lips OR price < 1d EMA(34)
            if (jaws[i] > teeth[i] and teeth[i] > lips[i]) or close[i] < lips[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish Alligator alignment OR price > lips OR price > 1d EMA(34)
            if (jaws[i] < teeth[i] and teeth[i] < lips[i]) or close[i] > lips[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals