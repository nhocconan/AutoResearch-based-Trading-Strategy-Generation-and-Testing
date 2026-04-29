#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips)
# Bullish when Lips > Teeth > Jaw (alligator eating up), Bearish when Lips < Teeth < Jaw (alligator eating down)
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume confirmation (>1.8x 40-period average) filters low-quality signals
# Works in bull/bear: volume confirms participation, 1d EMA50 filters whipsaws during ranges
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
    
    # Williams Alligator: three smoothed moving averages (SMMA)
    # SMMA is similar to EMA but with different smoothing
    close_series = pd.Series(close)
    
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = close_series.ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = close_series.ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, shifted 3 bars
    lips = close_series.ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
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
    
    start_idx = max(40, 13, 8, 5, 50)  # warmup for volume MA, Alligator lines, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
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
                # Bullish entry: Lips > Teeth > Jaw (alligator eating up) with price above 1d EMA50
                if curr_lips > curr_teeth and curr_teeth > curr_jaw and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Lips < Teeth < Jaw (alligator eating down) with price below 1d EMA50
                elif curr_lips < curr_teeth and curr_teeth < curr_jaw and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when alligator condition breaks (Lips <= Teeth or Teeth <= Jaw)
            if curr_lips <= curr_teeth or curr_teeth <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when alligator condition breaks (Lips >= Teeth or Teeth >= Jaw)
            if curr_lips >= curr_teeth or curr_teeth >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals