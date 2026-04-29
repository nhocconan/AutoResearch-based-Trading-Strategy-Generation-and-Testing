#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trending vs ranging markets
# Only trade when Alligator is "awake" (jaws, teeth, lips separated and aligned)
# Entry: price breaks Donchian(20) in direction of Alligator alignment + volume spike
# Exit: price crosses 8-period EMA or Donchian opposite break
# Designed for 6h timeframe with HTF 1d trend filter to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_Williams_Alligator_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (using 6h data)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Donchian channels for breakout
    def donchian_channel(high, low, period):
        """Calculate Donchian high and low"""
        dh = np.full_like(high, np.nan)
        dl = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            dh[i] = np.max(high[i-period+1:i+1])
            dl[i] = np.min(low[i-period+1:i+1])
        return dh, dl
    
    donchian_high, donchian_low = donchian_channel(high, low, 20)
    
    # ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation
    def volume_ma(volume, period):
        """Volume moving average"""
        ma = np.full_like(volume, np.nan)
        for i in range(period-1, len(volume)):
            ma[i] = np.mean(volume[i-period+1:i+1])
        return ma
    
    vol_ma_20 = volume_ma(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup period
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Alligator alignment check: "awake" and trending
        # Bullish: Lips > Teeth > Jaw (all aligned upward)
        # Bearish: Lips < Teeth < Jaw (all aligned downward)
        bullish_aligned = curr_lips > curr_teeth > curr_jaw
        bearish_aligned = curr_lips < curr_teeth < curr_jaw
        
        # Volume spike confirmation
        vol_spike = curr_vol > 2.0 * curr_vol_ma if curr_vol_ma > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: 
            # 1. Price crosses below 8-period EMA (teeth)
            # 2. Alligator loses bullish alignment
            # 3. Stoploss hit
            if (curr_close < curr_teeth or not bullish_aligned or curr_close < stop_price):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions:
            # 1. Price crosses above 8-period EMA (teeth)
            # 2. Alligator loses bearish alignment
            # 3. Stoploss hit
            if (curr_close > curr_teeth or not bearish_aligned or curr_close > stop_price):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND bullish Alligator alignment AND price > 1d EMA34 AND volume spike
            if (curr_close > curr_donchian_high and bullish_aligned and 
                curr_close > curr_ema_1d and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian low AND bearish Alligator alignment AND price < 1d EMA34 AND volume spike
            elif (curr_close < curr_donchian_low and bearish_aligned and 
                  curr_close < curr_ema_1d and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals