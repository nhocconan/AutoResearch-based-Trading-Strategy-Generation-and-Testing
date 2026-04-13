#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# Long: Lips > Teeth > Jaw (bullish alignment) + price above 1d EMA50 + volume > 1.5x avg volume
# Short: Lips < Teeth < Jaw (bearish alignment) + price below 1d EMA50 + volume > 1.5x avg volume
# Williams Alligator helps filter whipsaws in sideways markets.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in both bull and bear markets by using 1d EMA as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components (using 4h data)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA = smoothed moving average (similar to EMA but different smoothing)
    
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    for i in range(8, len(jaw)):
        jaw_shifted[i] = jaw[i-8]
    for i in range(5, len(teeth)):
        teeth_shifted[i] = teeth[i-5]
    for i in range(3, len(lips)):
        lips_shifted[i] = lips[i-3]
    
    # Average volume (20-period = 20*4h = 10 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        ema_trend = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Alligator alignment
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Long: bullish alignment + price above EMA50 + volume confirmation
            if (bullish_alignment and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: bearish alignment + price below EMA50 + volume confirmation
            elif (bearish_alignment and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment or price below EMA50
            if (bearish_alignment or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish alignment or price above EMA50
            if (bullish_alignment or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Williams_Alligator_EMA_Volume"
timeframe = "4h"
leverage = 1.0