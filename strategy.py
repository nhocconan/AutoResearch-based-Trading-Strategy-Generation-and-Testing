#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_alligator_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return signals
    
    # Calculate Williams Alligator on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Jaw (Blue): 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series((high_1d + low_1d) / 2).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    
    # Teeth (Red): 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series((high_1d + low_1d) / 2).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    
    # Lips (Green): 5-period SMMA, shifted 3 bars forward
    lips = pd.Series((high_1d + low_1d) / 2).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    # Convert to numpy arrays and handle NaN from shift
    jaw = jaw.fillna(0).values
    teeth = teeth.fillna(0).values
    lips = lips.fillna(0).values
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Trend strength filter: avoid weak trends
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / (atr_ma_50 + 1e-10)
    strong_trend = atr_ratio > 0.7
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or np.isnan(strong_trend[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        atr_val = atr[i]
        strong_trend_val = strong_trend[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma_20[i]
        
        # Alligator alignment conditions
        # Bullish alignment: Lips > Teeth > Jaw (green above red above blue)
        bullish_aligned = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish alignment: Lips < Teeth < Jaw (green below red below blue)
        bearish_aligned = lips_val < teeth_val and teeth_val < jaw_val
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        # Long: Bullish alignment + price above teeth + volume + strong trend
        if bullish_aligned and price_close > teeth_val and volume_confirmed and strong_trend_val:
            long_signal = True
        
        # Short: Bearish alignment + price below teeth + volume + strong trend
        if bearish_aligned and price_close < teeth_val and volume_confirmed and strong_trend_val:
            short_signal = True
        
        # Exit conditions: when alignment breaks
        exit_long = position == 1 and not bullish_aligned
        exit_short = position == -1 and not bearish_aligned
        
        # Stop loss conditions
        stop_long = position == 1 and price_low < (entry_price - 2.5 * atr_val)
        stop_short = position == -1 and price_high > (entry_price + 2.5 * atr_val)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            entry_price = price_close
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            entry_price = price_close
            signals[i] = -0.25
        elif position == 1 and (exit_long or stop_long):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or stop_short):
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6s Williams Alligator trend-following strategy with volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) on daily timeframe to identify trend direction and alignment.
# Enters long when Lips > Teeth > Jaw (bullish alignment) and price is above Teeth with volume confirmation.
# Enters short when Lips < Teeth < Jaw (bearish alignment) and price is below Teeth with volume confirmation.
# Includes ATR-based trend strength filter to avoid weak/choppy markets.
# Exits when Alligator alignment breaks or ATR stop loss (2.5x) is hit.
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by following established trends identified by the Alligator.