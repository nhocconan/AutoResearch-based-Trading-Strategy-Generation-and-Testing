#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA34 Trend + Volume Spike + ATR Stop
Hypothesis: Donchian breakouts capture strong momentum. 12h EMA34 filters for higher timeframe trend alignment.
Volume spike confirms breakout strength. Works in bull (long on upper band break + uptrend) and bear (short on lower band break + downtrend).
ATR-based stoploss manages risk. Target 20-40 trades/year on 4h to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period Donchian channels
    donchian_period = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    for i in range(donchian_period - 1, n):
        upper_band[i] = np.max(high[i-donchian_period+1:i+1])
        lower_band[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Calculate ATR(14) for stop management
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian, EMA34, ATR, volume MA
    start_idx = max(donchian_period, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_12h_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper band, volume confirmation, uptrend (price > EMA34)
            long_entry = (curr_close > upper_band[i]) and volume_confirm and (curr_close > ema_34_val)
            # Short: price breaks below lower band, volume confirmation, downtrend (price < EMA34)
            short_entry = (curr_close < lower_band[i]) and volume_confirm and (curr_close < ema_34_val)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below lower band OR 2.0*ATR trailing stop
            if curr_close < lower_band[i] or curr_close < (highest_since_entry - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above upper band OR 2.0*ATR trailing stop
            if curr_close > upper_band[i] or curr_close > (lowest_since_entry + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0