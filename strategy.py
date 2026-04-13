#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# The Williams Alligator uses three smoothed moving averages (Jaws, Teeth, Lips) to identify trends.
# In bull markets: Lips > Teeth > Jaws (bullish alignment)
# In bear markets: Lips < Teeth < Jaws (bearish alignment)
# The 1d trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation ensures trades have sufficient conviction.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator on 12h data
    # Jaws: 13-period SMMA, 8-bar shift
    # Teeth: 8-period SMMA, 5-bar shift  
    # Lips: 5-period SMMA, 3-bar shift
    def smoothed_mma(data, period):
        """Smoothed Moving Average (SMMA) - similar to Wilder's smoothing"""
        sma = np.full(len(data), np.nan)
        if len(data) < period:
            return sma
        # First value is simple average
        sma[period-1] = np.mean(data[:period])
        # Subsequent values: (prev_sma * (period-1) + current_price) / period
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaws = smoothed_mma(close, 13)
    teeth = smoothed_mma(close, 8)
    lips = smoothed_mma(close, 5)
    
    # Apply shifts (Jaws: 8 bars, Teeth: 5 bars, Lips: 3 bars)
    jaws_shifted = np.full_like(jaws, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    for i in range(8, len(jaws)):
        jaws_shifted[i] = jaws[i-8]
    for i in range(5, len(teeth)):
        teeth_shifted[i] = teeth[i-5]
    for i in range(3, len(lips)):
        lips_shifted[i] = lips[i-3]
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1d trend filter: EMA 34
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])  # First EMA value
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2/(34+1)) + (ema_34_1d[i-1] * (33)/(34+1))
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(34, n):  # Start after enough data for all indicators
        # Skip if any required data is not ready
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(jaws_shifted[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaws_val = jaws_shifted[i]
        ema_1d = ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaws AND price above 1d EMA
            if (lips_val > teeth_val and 
                teeth_val > jaws_val and 
                price > ema_1d and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Bearish alignment: Lips < Teeth < Jaws AND price below 1d EMA
            elif (lips_val < teeth_val and 
                  teeth_val < jaws_val and 
                  price < ema_1d and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment OR price crosses below 1d EMA
            if (lips_val < teeth_val or 
                teeth_val < jaws_val or
                price < ema_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish alignment OR price crosses above 1d EMA
            if (lips_val > teeth_val or 
                teeth_val > jaws_val or
                price > ema_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Williams_Alligator_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0