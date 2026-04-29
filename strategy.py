#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses Donchian channels for clear structure and breakout signals
# 1d EMA34 ensures alignment with higher timeframe trend
# Volume confirmation > 1.5x average to filter weak breakouts
# ATR-based stoploss for risk management
# Designed for moderate trade frequency (~25-35 trades/year) to minimize fee drag
# Works in both bull and bear markets by following higher timeframe trend

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period Donchian channels (using previous period to avoid look-ahead)
    # Highest high of last 20 periods
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Shift by 1 to use only completed periods
    highest_20_shifted = np.roll(highest_20, 1)
    lowest_20_shifted = np.roll(lowest_20, 1)
    highest_20_shifted[0] = np.nan
    lowest_20_shifted[0] = np.nan
    
    # Align Donchian levels to 4h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20_shifted)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20_shifted)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34, 14)  # Donchian, 1d EMA34, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_highest_20 = highest_20_aligned[i]
        curr_lowest_20 = lowest_20_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price breaks below Donchian low
            if curr_low <= entry_price - 2.0 * curr_atr or curr_close < curr_lowest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price breaks above Donchian high
            if curr_high >= entry_price + 2.0 * curr_atr or curr_close > curr_highest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above Donchian high, 1d EMA34 up-trend, volume confirmed
            if curr_high > curr_highest_20 and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when price breaks below Donchian low, 1d EMA34 down-trend, volume confirmed
            elif curr_low < curr_lowest_20 and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals