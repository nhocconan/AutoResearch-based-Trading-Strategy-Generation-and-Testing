#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA34 trend filter and volume spike confirmation
# Alligator uses SMAs: Jaw=13, Teeth=8, Lips=5. In trending markets, lines are ordered and separated.
# In bull trend (price > 1d EMA34): Lips > Teeth > Jaw -> long when Lips crosses above Teeth with volume
# In bear trend (price < 1d EMA34): Jaw > Teeth > Lips -> short when Jaw crosses below Teeth with volume
# Volume confirmation (>1.8x 20-period average) reduces false signals. Target ~12-37 trades/year on 12h.

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
    open_price = prices['open'].values
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator SMAs (on 12h data)
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth = close_s.rolling(window=8, min_periods=8).mean().values    # Teeth (8)
    lips = close_s.rolling(window=5, min_periods=5).mean().values     # Lips (5)
    
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
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Alligator lines converge (trend weakening)
            if curr_close < entry_price - 2.0 * curr_atr or (curr_lips <= curr_teeth):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Alligator lines converge (trend weakening)
            if curr_close > entry_price + 2.0 * curr_atr or (curr_jaw >= curr_teeth):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Bull trend: price > 1d EMA34, look for Lips crossing above Teeth
            if curr_close > curr_ema34_1d and curr_lips > curr_teeth and vol_confirm:
                # Additional confirmation: Lips was below or equal to Teeth previous bar (crossing up)
                if i > start_idx and curr_lips <= curr_teeth + 1e-9 and lips[i-1] <= teeth[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Bear trend: price < 1d EMA34, look for Jaw crossing below Teeth
            elif curr_close < curr_ema34_1d and curr_jaw < curr_teeth and vol_confirm:
                # Additional confirmation: Jaw was above or equal to Teeth previous bar (crossing down)
                if i > start_idx and curr_jaw >= curr_teeth - 1e-9 and jaw[i-1] >= teeth[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals