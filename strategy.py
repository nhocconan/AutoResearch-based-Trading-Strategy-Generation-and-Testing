#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend alignment; only trade when all three lines are aligned (bullish/bearish)
# 1d EMA50 ensures we only trade in direction of daily trend to avoid counter-trend whipsaws
# Volume spike (>2x 20-period average) confirms breakout strength
# Designed for ~12-25 trades/year on 12h timeframe to minimize fee drag while capturing strong trends
# Works in bull/bear markets via 1d EMA50 trend filter - only trades in direction of daily momentum

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    # SMMA = smoothed moving average (similar to EMA but with different alpha)
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines forward as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions that don't have valid data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 20  # volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_jaw = jaw_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Alligator lines lose bullish alignment
            if (curr_close < entry_price - 2.5 * atr_at_entry or 
                not (curr_lips > curr_teeth > curr_jaw)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Alligator lines lose bearish alignment
            if (curr_close > entry_price + 2.5 * atr_at_entry or 
                not (curr_lips < curr_teeth < curr_jaw)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_align = curr_lips > curr_teeth > curr_jaw
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_align = curr_lips < curr_teeth < curr_jaw
            
            # Long entry: bullish alignment + price above Jaw + 1d EMA50 uptrend + volume confirmation
            if bullish_align and curr_close > curr_jaw and curr_close > curr_ema50_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short entry: bearish alignment + price below Jaw + 1d EMA50 downtrend + volume confirmation
            elif bearish_align and curr_close < curr_jaw and curr_close < curr_ema50_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals