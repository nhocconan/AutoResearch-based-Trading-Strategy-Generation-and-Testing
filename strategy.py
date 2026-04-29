#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; 1d EMA50 ensures alignment with higher timeframe trend
# Volume confirmation (>1.8x 20-period average) reduces false breakouts
# Designed for ~20-50 trades/year to minimize fee drag while participating in established trends
# Works in bull/bear via 1d EMA50 trend filter - only trades in direction of 1d momentum
# Uses ATR-based stoploss (2.0x ATR) and time-based exit (10 bars) to manage risk

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
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
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_bar = 0
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit, time-based exit (10 bars), or price breaks below Donchian low
            if (curr_close < entry_price - 2.0 * curr_atr or 
                (i - entry_bar) >= 10 or 
                curr_close < curr_donchian_low):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit, time-based exit (10 bars), or price breaks above Donchian high
            if (curr_close > entry_price + 2.0 * curr_atr or 
                (i - entry_bar) >= 10 or 
                curr_close > curr_donchian_high):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long breakout when price closes above Donchian high with 1d EMA50 uptrend and volume confirmation
            if curr_close > curr_donchian_high and curr_close > curr_ema50_1d and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                entry_bar = i
            # Short breakout when price closes below Donchian low with 1d EMA50 downtrend and volume confirmation
            elif curr_close < curr_donchian_low and curr_close < curr_ema50_1d and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                entry_bar = i
            else:
                signals[i] = 0.0
    
    return signals