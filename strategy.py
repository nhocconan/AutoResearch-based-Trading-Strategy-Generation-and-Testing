#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above 20-period 12h Donchian high AND 1d EMA34 uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below 20-period 12h Donchian low AND 1d EMA34 downtrend AND volume > 1.5x 20-period median.
# Uses Donchian channels for structure, 1d EMA34 for higher-timeframe trend alignment, volume for conviction.
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag.

name = "12h_Donchian20_Breakout_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian(20) - get actual 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period high/low on 12h data
    donch_high_12h = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (primary timeframe)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Donchian and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price > Donchian high AND uptrend AND volume spike
            if curr_close > donch_high_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Price < Donchian low AND downtrend AND volume spike
            elif curr_close < donch_low_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price < Donchian low OR trend turns down
            if curr_close < donch_low_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price > Donchian high OR trend turns up
            if curr_close > donch_high_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals