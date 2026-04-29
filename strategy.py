#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trend and momentum.
# Long when Lips > Teeth > Jaw (bullish alignment) and price > 1d EMA50, with volume > 1.5x 20-period average.
# Short when Lips < Teeth < Jaw (bearish alignment) and price < 1d EMA50, with volume confirmation.
# Designed for ~15-30 trades/year on 12h timeframe to minimize fee drag while capturing strong trends.
# Works in both bull and bear markets via 1d trend filter - only trades in direction of higher timeframe trend.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h data (Jaw=13, Teeth=8, Lips=5 periods)
    # Smoothed moving average (SMMA) = EMA with alpha = 1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    lips = smma(close, 5)   # Lips: 5-period SMMA
    
    # Calculate 20-period average volume for confirmation (on 12h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: when Alligator alignment breaks (Lips < Teeth or Teeth < Jaw) or price < 1d EMA50
            if curr_lips < curr_teeth or curr_teeth < curr_jaw or curr_close < curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: when Alligator alignment breaks (Lips > Teeth or Teeth > Jaw) or price > 1d EMA50
            if curr_lips > curr_teeth or curr_teeth > curr_jaw or curr_close > curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: bullish Alligator alignment (Lips > Teeth > Jaw) and price > 1d EMA50
            if vol_confirm and curr_close > curr_ema50_1d:
                if curr_lips > curr_teeth and curr_teeth > curr_jaw:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
            # Short entry: bearish Alligator alignment (Lips < Teeth < Jaw) and price < 1d EMA50
            elif vol_confirm and curr_close < curr_ema50_1d:
                if curr_lips < curr_teeth and curr_teeth < curr_jaw:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals