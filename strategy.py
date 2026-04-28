# 12h Williams Alligator with Volume Confirmation and Trend Filter
# Strategy uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction
# Entry when price crosses above/below Alligator teeth with volume confirmation
# Exit when price crosses back through teeth or opposite signal
# Designed for 12h timeframe with 1d/1h trend filters to reduce whipsaw
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d data
    # Jaw (Blue): 13-period SMMA smoothed 8 periods ahead
    # Teeth (Red): 8-period SMMA smoothed 5 periods ahead  
    # Lips (Green): 5-period SMMA smoothed 3 periods ahead
    close_1d = df_1d['close'].values
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_CLOSE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Williams Alligator lines
    jaw = smma(close_1d, 13)  # 13-period SMMA
    teeth = smma(close_1d, 8)  # 8-period SMMA
    lips = smma(close_1d, 5)   # 5-period SMMA
    
    # Shift jaws forward by 8, teeth by 5, lips by 3 (as per Williams Alligator)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Get 1h trend filter (EMA 50)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    # Volume confirmation (20-period volume MA spike)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1h_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator signals
        # Bullish: Lips > Teeth > Jaw (all aligned and separated)
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Bearish: Lips < Teeth < Jaw (all aligned and separated)
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Price position relative to teeth (trigger line)
        price_above_teeth = close[i] > teeth_aligned[i]
        price_below_teeth = close[i] < teeth_aligned[i]
        
        # Trend filter: 1h EMA50 direction
        uptrend_filter = close[i] > ema_50_1h_aligned[i]
        downtrend_filter = close[i] < ema_50_1h_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry conditions
        long_entry = bullish_alignment and price_above_teeth and uptrend_filter and vol_confirm
        short_entry = bearish_alignment and price_below_teeth and downtrend_filter and vol_confirm
        
        # Exit conditions: price crosses back through teeth or opposite alignment
        long_exit = price_below_teeth or bearish_alignment
        short_exit = price_above_teeth or bullish_alignment
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_Teeth_Cross_Volume"
timeframe = "12h"
leverage = 1.0