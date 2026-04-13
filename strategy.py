#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA trend filter and volume confirmation.
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# When the three lines are intertwined, the market is ranging (Alligator sleeping).
# When they diverge, a trend is forming (Alligator waking up and feeding).
# Combined with 1-week EMA trend filter and volume spikes, it avoids whipsaws and trades with momentum.
# Works in both bull and bear markets by taking long signals only when Alligator is bullish (Lips > Teeth > Jaw) 
# and price above 1w EMA, and short when bearish (Lips < Teeth < Jaw) and price below 1w EMA.
# Target: 12-37 trades per year (50-150 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-week EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.zeros(len(close_1w))
    ema_multiplier50 = 2 / (50 + 1)
    ema50_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema50_1w[i] = (close_1w[i] - ema50_1w[i-1]) * ema_multiplier50 + ema50_1w[i-1]
    
    # Calculate Williams Alligator on daily timeframe
    # Jaw: Blue line, 13-period SMMA shifted 8 bars ahead
    # Teeth: Red line, 8-period SMMA shifted 5 bars ahead  
    # Lips: Green line, 5-period SMMA shifted 3 bars ahead
    close_1d = df_1d['close'].values
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(data, period):
        sma = np.full(len(data), np.nan)
        if len(data) < period:
            return sma
        # First value is simple average
        sma[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    smma_13 = smma(close_1d, 13)
    smma_8 = smma(close_1d, 8)
    smma_5 = smma(close_1d, 5)
    
    # Shift the SMMA lines as per Alligator specification
    jaw = np.full_like(smma_13, np.nan)
    teeth = np.full_like(smma_8, np.nan)
    lips = np.full_like(smma_5, np.nan)
    
    # Jaw: 13-period SMMA shifted 8 bars ahead
    if len(smma_13) > 8:
        jaw[8:] = smma_13[:-8]
    # Teeth: 8-period SMMA shifted 5 bars ahead
    if len(smma_8) > 5:
        teeth[5:] = smma_8[:-5]
    # Lips: 5-period SMMA shifted 3 bars ahead
    if len(smma_5) > 3:
        lips[3:] = smma_5[:-3]
    
    # Align all indicators to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate average volume (24-period = 12 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1w_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Alligator conditions:
        # Bullish: Lips > Teeth > Jaw (all lines aligned upward)
        # Bearish: Lips < Teeth < Jaw (all lines aligned downward)
        bullish_alligator = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bearish_alligator = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        if position == 0:
            # Long: Bullish Alligator + above weekly EMA50 + volume confirmation
            if (bullish_alligator and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bearish Alligator + below weekly EMA50 + volume confirmation
            elif (bearish_alligator and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish or trend turns down
            if (not bullish_alligator or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator turns bullish or trend turns up
            if (not bearish_alligator or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_WilliamsAlligator_Trend_Volume"
timeframe = "12h"
leverage = 1.0