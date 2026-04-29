#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based trailing stop
# Donchian channels provide clear structure for breakouts; daily EMA filter ensures trades align with higher timeframe momentum
# ATR trailing stop (3x) protects gains during reversals while allowing trends to run
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Works in bull/bear via 1d EMA34 trend filter - only trades in direction of daily momentum
# Uses volume confirmation (>1.8x 20-period average) to reduce false breakouts

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR (14-period) for stoploss and position sizing
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        # Handle position exits and trailing stops
        if position == 1:  # Long position
            # Update highest price since entry for trailing stop
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: stoploss hit or trailing stop hit
            if (curr_close < entry_price - 2.0 * curr_atr or  # initial stop
                curr_close < highest_since_entry - 3.0 * curr_atr):  # trailing stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest price since entry for trailing stop
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: stoploss hit or trailing stop hit
            if (curr_close > entry_price + 2.0 * curr_atr or  # initial stop
                curr_close > lowest_since_entry + 3.0 * curr_atr):  # trailing stop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long breakout when price closes above upper Donchian with 1d EMA34 uptrend and volume confirmation
            if curr_close > curr_highest_high and curr_close > curr_ema34_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close  # reset for potential flip
            # Short breakout when price closes below lower Donchian with 1d EMA34 downtrend and volume confirmation
            elif curr_close < curr_lowest_low and curr_close < curr_ema34_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
                highest_since_entry = curr_close  # reset for potential flip
            else:
                signals[i] = 0.0
    
    return signals