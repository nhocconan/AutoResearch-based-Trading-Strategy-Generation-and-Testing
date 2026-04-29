#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses 6h Donchian channels for breakout entries in the direction of 1d EMA34 trend
# Volume confirmation (>1.8x 30-period average) filters for institutional participation
# Trend filter uses 1d EMA34 to avoid counter-trend trades in both bull and bear markets
# Target: 75-150 total trades over 4 years (19-37/year) to balance edge and fee drag
# Designed for 6h timeframe to capture swings with controlled frequency
# BTC/ETH focus: requires EMA alignment and volume confirmation to avoid SOL-only bias

name = "6h_Donchian20_1dEMA34_VolumeSpike_Trend"
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
    
    # Get 1d data for EMA34 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Donchian channels (20-period)
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 30-period average volume for confirmation
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(34, 20, 30, 14)  # EMA34, Donchian, volume MA, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_rolling[i]) or np.isnan(low_rolling[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_upper = high_rolling[i]
        curr_lower = low_rolling[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_30[i]
        
        # Handle stoploss and exits
        if position == 1:  # Long position
            # Stoploss: price closes below entry - 2.5 * ATR_at_entry
            if curr_close < entry_price - 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below lower Donchian or trend turns down
            elif curr_close < curr_lower or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: price closes above entry + 2.5 * ATR_at_entry
            if curr_close > entry_price + 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above upper Donchian or trend turns up
            elif curr_close > curr_upper or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 30-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long entry: price breaks above upper Donchian in uptrend (price > EMA34_1d)
            if vol_confirm and curr_close > curr_ema34_1d:
                if curr_high > curr_upper:  # Break above upper Donchian
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            # Short entry: price breaks below lower Donchian in downtrend (price < EMA34_1d)
            elif vol_confirm and curr_close < curr_ema34_1d:
                if curr_low < curr_lower:  # Break below lower Donchian
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals