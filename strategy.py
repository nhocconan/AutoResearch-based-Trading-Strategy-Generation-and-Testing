#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
# Long when price breaks above 6h Donchian upper (20) AND weekly pivot shows bullish bias (price > weekly VWAP) AND volume > 1.5x 20-period median.
# Short when price breaks below 6h Donchian lower (20) AND weekly pivot shows bearish bias (price < weekly VWAP) AND volume > 1.5x 20-period median.
# Exit on ATR trailing stop (2.5x ATR) or opposite Donchian breakout.
# Uses weekly structure for major trend bias, Donchian for breakout timing, volume for confirmation.
# Target: 12-30 trades/year on 6h timeframe. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm_ATR_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly VWAP for trend bias (loaded once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Typical price for VWAP
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # VWAP = sum(price * volume) / sum(volume)
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w_values = vwap_1w.values
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w_values)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 20-period ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for Donchian, ATR, and volume median
    start_idx = max(lookback, 20) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(vwap_1w_aligned[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Donchian breakout conditions
        breakout_up = curr_close > highest_high[i]  # break above upper band
        breakout_down = curr_close < lowest_low[i]  # break below lower band
        
        # Weekly pivot bias: price > weekly VWAP = bullish, price < weekly VWAP = bearish
        weekly_bullish = curr_close > vwap_1w_aligned[i]
        weekly_bearish = curr_close < vwap_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND weekly bullish bias AND volume spike
            if breakout_up and weekly_bullish and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Breakout down AND weekly bearish bias AND volume spike
            elif breakout_down and weekly_bearish and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry for trailing stop
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR trailing stop OR Donchian breakout down
            stop_price = highest_since_entry - 2.5 * curr_atr
            if curr_close < stop_price or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry for trailing stop
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR trailing stop OR Donchian breakout up
            stop_price = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stop_price or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals