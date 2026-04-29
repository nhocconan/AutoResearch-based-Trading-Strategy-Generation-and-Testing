#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period SMMA shifted 8) and Lips > Teeth > Jaw (bullish alignment)
# Short when price < Alligator Jaw and Lips < Teeth < Jaw (bearish alignment)
# Volume confirmation: current volume > 1.5x 20-period average
# Exit when Alligator lines cross (Lips crosses Teeth) or price crosses Jaw
# Uses discrete position sizing (0.25) to target 12-30 trades/year on 12h timeframe.
# Williams Alligator identifies trends effectively in both bull and bear markets by showing
# when the market is sleeping (all lines intertwined) vs waking up (lines diverging).

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator (SMMA = Smoothed Moving Average)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars  
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_raw = smma(median_price_12h, 13)
    teeth_raw = smma(median_price_12h, 8)
    lips_raw = smma(median_price_12h, 5)
    
    # Shift the lines: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw_shifted = np.full_like(jaw_raw, np.nan)
    teeth_shifted = np.full_like(teeth_raw, np.nan)
    lips_shifted = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > 8:
        jaw_shifted[8:] = jaw_raw[:-8]
    if len(teeth_raw) > 5:
        teeth_shifted[5:] = teeth_raw[:-5]
    if len(lips_raw) > 3:
        lips_shifted[3:] = lips_raw[:-3]
    
    # Align 12h Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_shifted)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Volume and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_jaw = jaw_aligned[i]
        curr_teeth = teeth_aligned[i]
        curr_lips = lips_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Lips crosses below Teeth (bullish alignment broken) OR price below Jaw
            if curr_lips < curr_teeth or curr_close < curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Lips crosses above Teeth (bearish alignment broken) OR price above Jaw
            if curr_lips > curr_teeth or curr_close > curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = curr_lips > curr_teeth and curr_teeth > curr_jaw
            # Bearish alignment: Lips < Teeth < Jaw  
            bearish_alignment = curr_lips < curr_teeth and curr_teeth < curr_jaw
            
            # Long when bullish alignment, price above Jaw, and volume confirmed
            if bullish_alignment and curr_close > curr_jaw and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when bearish alignment, price below Jaw, and volume confirmed
            elif bearish_alignment and curr_close < curr_jaw and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals