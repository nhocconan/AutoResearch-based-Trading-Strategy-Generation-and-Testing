#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1-day EMA50 trend filter and volume confirmation
# Williams Alligator consists of three SMAs: Jaw (13), Teeth (8), Lips (5)
# In trending markets: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
# In ranging markets: lines intertwine
# We add 1-day EMA50 filter to ensure we only trade in direction of higher timeframe trend
# Volume confirmation (1.5x 20-period average) adds conviction to breakouts
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag on 6h timeframe
# Williams Alligator catches trends early, daily EMA50 filter avoids counter-trend trades, volume confirmation adds conviction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1-day EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Williams Alligator on 6h (using close prices) ===
    # Jaw: 13-period SMMA (smoothed moving average)
    # Teeth: 8-period SMMA
    # Lips: 5-period SMMA
    # SMMA calculation: SMMA(t) = (SMMA(t-1) * (period-1) + close(t)) / period
    def smma(series, period):
        result = np.full_like(series, np.nan)
        if len(series) >= period:
            # First value is SMA
            result[period-1] = np.mean(series[:period])
            # Subsequent values
            for i in range(period, len(series)):
                result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # === 6h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_val = ema_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume for confirmation
        
        # Williams Alligator signals:
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        
        # Exit conditions: when alignment breaks or contrary to higher timeframe trend
        if position == 1:  # Long position
            # Exit if: bearish alignment OR price below EMA50 (contrary to trend)
            if not (lips_val > teeth_val > jaw_val) or price < ema_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit if: bullish alignment OR price above EMA50 (contrary to trend)
            if not (lips_val < teeth_val < jaw_val) or price > ema_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions (only when flat)
        if position == 0:
            # Bullish alignment AND price above EMA50 AND volume confirmation
            if lips_val > teeth_val > jaw_val and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Bearish alignment AND price below EMA50 AND volume confirmation
            elif lips_val < teeth_val < jaw_val and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Volume1.5x"
timeframe = "6h"
leverage = 1.0