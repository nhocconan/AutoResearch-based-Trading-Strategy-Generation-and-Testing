#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator system with 1d trend filter and volume confirmation.
# Uses Alligator's Jaw (TEETH=13), Teeth (TEETH=8), Lips (LIPS=5) smoothed with SMMA.
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 2.0x 20-bar avg.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 2.0x 20-bar avg.
# Exit when Alligator lines intertwine (Lips crosses Teeth or Jaw) indicating loss of momentum.
# Williams Alligator is effective in both trending and ranging markets, and the 1d EMA50 filter
# ensures we only trade with the higher timeframe trend, reducing false signals in chop.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMMA (Smoothed Moving Average) of median price
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # SMMA calculation: similar to EMA but with different smoothing
    # Jaw: SMMA(13, 8) - 13 period, 8 future shift
    # Teeth: SMMA(8, 5) - 8 period, 5 future shift  
    # Lips: SMMA(5, 3) - 5 period, 3 future shift
    def smma(data, period, shift):
        """Smoothed Moving Average with shift"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        # First value is SMA
        sma = np.full_like(data, np.nan)
        sma[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA(i) = (SMMA(i-1) * (period-1) + data[i]) / period
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        # Apply shift (Williams Alligator uses future shift to align with current bar)
        if shift > 0:
            sma = np.roll(sma, -shift)
            sma[-shift:] = np.nan
        return sma
    
    jaw = smma(median_price, 13, 8)
    teeth = smma(median_price, 8, 5)
    lips = smma(median_price, 5, 3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = curr_lips > curr_teeth and curr_teeth > curr_jaw
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = curr_lips < curr_teeth and curr_teeth < curr_jaw
            
            # Long: bullish alignment, price above 1d EMA50, volume spike
            if (bullish_alignment and 
                close[i] > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, price below 1d EMA50, volume spike
            elif (bearish_alignment and 
                  close[i] < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: loss of bullish alignment (Lips crosses below Teeth or Jaw)
            if curr_lips <= curr_teeth or curr_lips <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: loss of bearish alignment (Lips crosses above Teeth or Jaw)
            if curr_lips >= curr_teeth or curr_lips >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals