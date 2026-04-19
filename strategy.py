#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with daily trend filter and volume confirmation.
# Long when price > Alligator Jaw and Alligator Mouth opens upward (Jaw < Teeth < Lips) with price above 1d EMA50 and volume spike (>1.8x average).
# Short when price < Alligator Jaw and Alligator Mouth opens downward (Jaw > Teeth > Lips) with price below 1d EMA50 and volume spike.
# Williams Alligator uses smoothed SMAs (Jaw: 13-period SMMA shifted 8, Teeth: 8-period SMMA shifted 5, Lips: 5-period SMMA shifted 3).
# The 1d EMA50 filter ensures trend alignment, reducing whipsaw in sideways markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 12-37 trades/year per symbol (~50-150 total over 4 years).
name = "12h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=np.float64)
    result = np.full_like(data, np.nan, dtype=np.float64)
    # First value is SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (wait for daily close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator components on 12h data
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = smma(close, 13)
    jaw = np.roll(jaw, 8)  # Shift right by 8
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = smma(close, 8)
    teeth = np.roll(teeth, 5)  # Shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = smma(close, 5)
    lips = np.roll(lips, 3)  # Shift right by 3
    lips[:3] = np.nan
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 13+8, 8+5, 5+3)  # Need all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_trend = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        
        # Alligator Mouth direction
        mouth_open_up = jaw_val < teeth_val < lips_val  # Jaw < Teeth < Lips
        mouth_open_down = jaw_val > teeth_val > lips_val  # Jaw > Teeth > Lips
        
        if position == 0:
            # Enter long: price > Jaw AND Mouth opens upward AND price > EMA50 AND volume confirmed
            if price > jaw_val and mouth_open_up and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Jaw AND Mouth opens downward AND price < EMA50 AND volume confirmed
            elif price < jaw_val and mouth_open_down and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price < Jaw or Mouth closes or price < EMA50
            if price < jaw_val or not mouth_open_up or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price > Jaw or Mouth closes or price > EMA50
            if price > jaw_val or not mouth_open_down or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals