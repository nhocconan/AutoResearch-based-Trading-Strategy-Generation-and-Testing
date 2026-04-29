#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation
# Long: price breaks above Donchian(20) high + price > 12h EMA50 + volume > 2.0x 20-period average
# Short: price breaks below Donchian(20) low + price < 12h EMA50 + volume > 2.0x 20-period average
# Exit: ATR-based trailing stop (3x ATR) or opposite Donchian breakout
# Designed for ~25-40 trades/year to minimize fee drag while capturing strong trends
# Works in bull/bear via 12h EMA50 trend filter - only trades with higher timeframe momentum

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, 20, 14, 20)  # EMA, Donchian, ATR, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_highest_high_20 = highest_high_20[i]
        curr_lowest_low_20 = lowest_low_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: trailing stoploss hit or opposite Donchian breakout
            if curr_close < highest_since_entry - 3.0 * curr_atr or curr_low < curr_lowest_low_20:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: trailing stoploss hit or opposite Donchian breakout
            if curr_close > lowest_since_entry + 3.0 * curr_atr or curr_high > curr_highest_high_20:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above Donchian(20) high with 12h EMA50 uptrend and volume confirmation
            if curr_high > curr_highest_high_20 and curr_close > curr_ema50_12h and vol_confirm:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                highest_since_entry = curr_high
            # Short when price breaks below Donchian(20) low with 12h EMA50 downtrend and volume confirmation
            elif curr_low < curr_lowest_low_20 and curr_close < curr_ema50_12h and vol_confirm:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals