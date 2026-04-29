#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation
# Williams Alligator uses smoothed SMAs: Jaw=13-period SMMA(8), Teeth=8-period SMMA(5), Lips=5-period SMMA(3)
# Bullish when Lips > Teeth > Jaw (all aligned upward), Bearish when Lips < Teeth < Jaw (all aligned downward)
# 1d EMA50 filter ensures alignment with daily trend to avoid counter-trend trades during ranges
# Volume confirmation (>1.8x 40-period average) filters low-quality breakouts
# Works in bull/bear: Alligator identifies trending regimes, volume confirms participation, 1d EMA50 filters whipsaws
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_WilliamsAlligator_VolumeConfirm_1dEMA50_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate SMMA (Smoothed Moving Average) - same as RMA/Wilder's MA
    def smma(source, period):
        """Smoothed Moving Average (same as RMA/Wilder's MA)"""
        if len(source) < period:
            return np.full(len(source), np.nan)
        result = np.full(len(source), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Williams Alligator components
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Shift Alligator lines by future offset (8,5,3 respectively) as per original formula
    # Jaw offset 8, Teeth offset 5, Lips offset 3
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set first values to NaN due to roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.8x 40-period average
    vol_ma_40 = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    volume_confirm = volume > (1.8 * vol_ma_40)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(40, 13, 8, 5, 50)  # warmup for volume MA, Alligator, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(vol_ma_40[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: Lips > Teeth > Jaw (all aligned upward) with price above 1d EMA50
                if curr_lips > curr_teeth > curr_jaw and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Lips < Teeth < Jaw (all aligned downward) with price below 1d EMA50
                elif curr_lips < curr_teeth < curr_jaw and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when Alligator loses bullish alignment (Lips <= Teeth or Teeth <= Jaw)
            if curr_lips <= curr_teeth or curr_teeth <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Alligator loses bearish alignment (Lips >= Teeth or Teeth >= Jaw)
            if curr_lips >= curr_teeth or curr_teeth >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals