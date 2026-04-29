#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with 1w EMA50 trend filter and volume confirmation
# Williams Alligator uses smoothed medians: Jaw=EMA13(8), Teeth=EMA8(5), Lips=EMA5(3) of median price
# In bull markets (price > 1w EMA50), go long when Lips > Teeth > Jaw (bullish alignment)
# In bear markets (price < 1w EMA50), go short when Lips < Teeth < Jaw (bearish alignment)
# Volume confirmation (>1.5x 20-period average) filters weak breakouts
# Designed for ~7-25 trades/year on 1d timeframe to minimize fee drag while capturing major trends
# Works in both bull and bear via 1w EMA50 trend filter - only trades in direction of higher timeframe momentum

name = "1d_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
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
    
    # Get 1w data for EMA50 trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator components (on 1d data)
    median_price = (high + low) / 2.0
    median_s = pd.Series(median_price)
    
    # Jaw: EMA13 of median, smoothed 8 periods
    jaw = median_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: EMA8 of median, smoothed 5 periods
    teeth = median_s.ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: EMA5 of median, smoothed 3 periods
    lips = median_s.ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values
    
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
    
    start_idx = 30  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if curr_close < entry_price - 2.0 * curr_atr or curr_lips <= curr_teeth or curr_teeth <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if curr_close > entry_price + 2.0 * curr_atr or curr_lips >= curr_teeth or curr_teeth >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = curr_lips > curr_teeth and curr_teeth > curr_jaw
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = curr_lips < curr_teeth and curr_teeth < curr_jaw
            
            # Long entry when price > 1w EMA50 (bullish regime) AND bullish alignment with volume confirmation
            if curr_close > curr_ema50_1w and bullish_alignment and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry when price < 1w EMA50 (bearish regime) AND bearish alignment with volume confirmation
            elif curr_close < curr_ema50_1w and bearish_alignment and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals