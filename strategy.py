#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike
# Uses Donchian channel breakout for trend continuation with volume confirmation
# Trend filter uses 1d EMA34 to avoid counter-trend trades in both bull and bear markets
# Volume confirmation (>2.0x 24-period average) ensures institutional participation
# ATR-based stoploss and trailing exit for risk management
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Designed for 12h timeframe to capture swings with controlled frequency
# BTC/ETH focus: requires EMA alignment and volume confirmation to avoid SOL-only bias

name = "12h_Donchian20_1dEMA34_VolumeConfirmation_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    # Highest high over last 20 periods
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low over last 20 periods
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 24-period average volume for confirmation
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    highest_since_entry = 0.0  # For trailing stop
    
    start_idx = max(34, 20, 24, 14)  # EMA34, Donchian, volume MA, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_24[i]
        
        # Handle position management
        if position == 1:  # Long position
            # Update highest high since entry for trailing stop
            highest_since_entry = max(highest_since_entry, curr_high)
            # Trailing stop: price closes below highest - 3.0 * ATR
            if curr_close < highest_since_entry - 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            # Exit: price breaks below Donchian low or trend turns down
            elif curr_close < curr_lowest_20 or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Trailing stop: price closes above lowest + 3.0 * ATR
            if curr_close > lowest_since_entry + 3.0 * curr_atr:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            # Exit: price breaks above Donchian high or trend turns up
            elif curr_close > curr_highest_20 or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 24-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long entry: price breaks above Donchian high in uptrend (price > EMA34_1d)
            if vol_confirm and curr_close > curr_ema34_1d:
                if curr_high > curr_highest_20:  # Break above Donchian high
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
                    highest_since_entry = curr_high
            # Short entry: price breaks below Donchian low in downtrend (price < EMA34_1d)
            elif vol_confirm and curr_close < curr_ema34_1d:
                if curr_low < curr_lowest_20:  # Break below Donchian low
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
                    lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals