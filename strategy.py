#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high and price > 1d EMA50 (uptrend)
# Short when price breaks below Donchian(20) low and price < 1d EMA50 (downtrend)
# Volume confirmation (>1.5x 20-period average) filters weak breakouts
# ATR(14) stoploss (2.0x ATR) manages risk
# Designed for ~25-35 trades/year on 4h timeframe to minimize fee drag
# Works in both bull and bear markets via 1d trend filter - only trades in trend direction

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Donchian, volume MA and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price breaks below Donchian low
            if curr_close < entry_price - 2.0 * curr_atr or curr_close < curr_lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price breaks above Donchian high
            if curr_close > entry_price + 2.0 * curr_atr or curr_close > curr_highest_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirm = curr_volume > 1.5 * curr_vol_ma
            
            # Long entry: price breaks above Donchian high in uptrend
            if vol_confirm and curr_close > curr_highest_high and curr_close > curr_ema50_1d:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian low in downtrend
            elif vol_confirm and curr_close < curr_lowest_low and curr_close < curr_ema50_1d:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals