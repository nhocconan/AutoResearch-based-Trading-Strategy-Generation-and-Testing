#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation.
# Alligator uses three SMAs (Jaw, Teeth, Lips) to identify trends and ranges.
# In strong trends, the SMAs diverge (mouth open); in ranges, they converge (mouth closed).
# Combined with 1d trend filter and volume spikes, it filters false signals.
# Target: 12-37 trades per year (50-150 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend filter
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier + ema50_1d[i-1]
    
    # Align 1d EMA to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw: SMA(13, 8) - 13-period SMA shifted 8 bars forward
    # Teeth: SMA(8, 5) - 8-period SMA shifted 5 bars forward
    # Lips: SMA(5, 3) - 5-period SMA shifted 3 bars forward
    jaw = np.full(n, np.nan)
    teeth = np.full(n, np.nan)
    lips = np.full(n, np.nan)
    
    # Calculate SMAs
    def calculate_sma(data, period):
        sma = np.full(len(data), np.nan)
        if len(data) < period:
            return sma
        sma[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            sma[i] = sma[i-1] + (data[i] - data[i-period]) / period
        return sma
    
    sma13 = calculate_sma(close, 13)
    sma8 = calculate_sma(close, 8)
    sma5 = calculate_sma(close, 5)
    
    # Shift SMAs to create Alligator lines
    for i in range(8, n):
        jaw[i] = sma13[i-8] if i-8 >= 0 and not np.isnan(sma13[i-8]) else np.nan
    for i in range(5, n):
        teeth[i] = sma8[i-5] if i-5 >= 0 and not np.isnan(sma8[i-5]) else np.nan
    for i in range(3, n):
        lips[i] = sma5[i-3] if i-3 >= 0 and not np.isnan(sma5[i-3]) else np.nan
    
    # Average volume (20-period = 10 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        
        # Alligator conditions
        # Mouth open (trending): Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
        # Mouth closed (ranging): lines intertwined
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirm = vol > 1.8 * avg_vol
        
        if position == 0:
            # Long: Uptrend (Lips > Teeth > Jaw) + above 1d EMA50 + volume confirmation
            if (lips_val > teeth_val and teeth_val > jaw_val and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Downtrend (Lips < Teeth < Jaw) + below 1d EMA50 + volume confirmation
            elif (lips_val < teeth_val and teeth_val < jaw_val and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Trend changes to downtrend or price breaks below 1d EMA
            if (lips_val < teeth_val or teeth_val < jaw_val or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Trend changes to uptrend or price breaks above 1d EMA
            if (lips_val > teeth_val or teeth_val > jaw_val or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_WilliamsAlligator_Trend_Volume"
timeframe = "12h"
leverage = 1.0