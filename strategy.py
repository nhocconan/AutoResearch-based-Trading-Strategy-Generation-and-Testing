#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses Donchian channels from 4h price action for breakout signals
# 1d EMA34 provides strong trend filter to avoid counter-trend trades
# Volume > 1.8x 20-period average confirms breakout strength
# ATR-based stoploss (2.0 ATR) and Camarilla H3/L3 profit targets
# Discrete position sizing (0.30) with trailing stop via signal=0 when stopped
# Target: ~25-35 trades/year to minimize fee drag while capturing strong trends
# Works in bull/bear via trend filter - only trades in direction of 1d EMA34
# Focus on BTC/ETH with proven Donchian structure + volume confirmation edge

name = "4h_Donchian20_1dEMA34_VolumeConfirm_TrendFilter"
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
    
    # Calculate ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Camarilla levels for profit targets
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    h3_1d = close_1d + range_1d * 1.1 / 6.0
    l3_1d = close_1d - range_1d * 1.1 / 6.0
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 34, 14)  # Donchian, EMA34, ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        curr_h3 = h3_1d_aligned[i]
        curr_l3 = l3_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            
            # Check stoploss: 2.0 ATR below highest since entry
            if curr_close < highest_since_entry - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            # Check profit target: Camarilla H3
            elif curr_close >= curr_h3:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Check stoploss: 2.0 ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            # Check profit target: Camarilla L3
            elif curr_close <= curr_l3:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 1.8x 20-period average
            vol_spike = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above Donchian upper band with 1d EMA34 uptrend and volume spike
            if curr_high > curr_highest_20 and curr_close > curr_ema34_1d and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
            # Short when price breaks below Donchian lower band with 1d EMA34 downtrend and volume spike
            elif curr_low < curr_lowest_20 and curr_close < curr_ema34_1d and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals