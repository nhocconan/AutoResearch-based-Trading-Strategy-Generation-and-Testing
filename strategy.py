#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator uses three SMAs (jaw=13, teeth=8, lips=5) to identify trends
# In bull markets (price > 1w EMA50), we go long when Alligator is bullish (lips > teeth > jaw) with volume confirmation
# In bear markets (price < 1w EMA50), we go short when Alligator is bearish (lips < teeth < jaw) with volume confirmation
# Volume confirmation: current volume > 1.5x 20-period average to reduce false signals
# Designed for ~12-37 trades/year on 12h timeframe to minimize fee drag while capturing Alligator trend signals
# Works in both bull and bear via 1w EMA50 trend filter - only trades in direction of higher timeframe momentum

name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike_v1"
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
    
    # Get 1w data for EMA50 trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator SMAs (on 12h data)
    close_s = pd.Series(close)
    # Jaw: 13-period SMA, Teeth: 8-period SMA, Lips: 5-period SMA
    jaw = close_s.rolling(window=13, min_periods=13).mean().values
    teeth = close_s.rolling(window=8, min_periods=8).mean().values
    lips = close_s.rolling(window=5, min_periods=5).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 20  # volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Alligator turns bearish (lips < teeth or teeth < jaw)
            if curr_close < entry_price - 2.0 * curr_atr or curr_lips < curr_teeth or curr_teeth < curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Alligator turns bullish (lips > teeth or teeth > jaw)
            if curr_close > entry_price + 2.0 * curr_atr or curr_lips > curr_teeth or curr_teeth > curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry when price > 1w EMA50 (bullish regime) AND Alligator is bullish with volume confirmation
            if curr_close > curr_ema50_1w and curr_lips > curr_teeth and curr_teeth > curr_jaw and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry when price < 1w EMA50 (bearish regime) AND Alligator is bearish with volume confirmation
            elif curr_close < curr_ema50_1w and curr_lips < curr_teeth and curr_teeth < curr_jaw and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals