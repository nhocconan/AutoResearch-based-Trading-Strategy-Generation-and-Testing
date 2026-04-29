#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses 1d timeframe for lower trade frequency (~10-30 trades/year) and Donchian channels for structure
# 1w EMA34 ensures alignment with weekly trend to avoid counter-trend trades
# Volume confirmation > 1.5x average to filter weak breakouts
# Discrete position sizing (0.25) and ATR-based stoploss (2.0 ATR) for risk control
# Designed to work in both bull and bear markets by following the weekly trend

name = "1d_Donchian20_1wEMA34_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
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
    
    start_idx = max(20, 34, 14)  # Volume, 1w EMA34, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Calculate Donchian channels (20-period)
        # Lookback 20 periods, excluding current bar to avoid look-ahead
        lookback_start = max(0, i - 20)
        lookback_end = i  # exclusive, so we use [lookback_start:i]
        if lookback_end - lookback_start < 20:
            signals[i] = 0.0
            continue
            
        highest_high = np.max(high[lookback_start:lookback_end])
        lowest_low = np.min(low[lookback_start:lookback_end])
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Donchian breakdown
            if curr_low <= entry_price - 2.0 * curr_atr or curr_close < lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Donchian breakout
            if curr_high >= entry_price + 2.0 * curr_atr or curr_close > highest_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above Donchian high, 1w EMA34 up-trend, volume confirmed
            if curr_high > highest_high and curr_close > curr_ema34_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when price breaks below Donchian low, 1w EMA34 down-trend, volume confirmed
            elif curr_low < lowest_low and curr_close < curr_ema34_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals