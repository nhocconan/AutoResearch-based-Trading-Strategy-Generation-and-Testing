#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
# Donchian channels capture established trends; breakouts with volume and 1w EMA50 alignment ride momentum
# Designed for ~7-25 trades/year to minimize fee drag while participating in strong trends
# Works in bull/bear via 1w EMA50 trend filter - only trades in direction of weekly momentum
# Uses strict volume confirmation (>2.0x 20-period average) to reduce false breakouts
# Exits on 2.0x ATR stoploss or when price retests the broken Donchian level

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
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
    
    # Get 1w data for EMA50 trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR (14-period) for stoploss
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
    atr_at_entry = 0.0
    breakout_level = 0.0  # Track the broken Donchian level for exit
    
    start_idx = 50  # Donchian and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        curr_highest = highest_high[i]
        curr_lowest = lowest_low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price retests broken lower Donchian level
            if curr_close < entry_price - 2.0 * curr_atr or curr_close < breakout_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price retests broken upper Donchian level
            if curr_close > entry_price + 2.0 * curr_atr or curr_close > breakout_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long breakout when price closes above upper Donchian with 1w EMA50 uptrend and volume confirmation
            if curr_close > curr_highest and curr_close > curr_ema50_1w and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
                breakout_level = curr_lowest  # Exit if price retests the broken lower level
            # Short breakout when price closes below lower Donchian with 1w EMA50 downtrend and volume confirmation
            elif curr_close < curr_lowest and curr_close < curr_ema50_1w and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
                breakout_level = curr_highest  # Exit if price retests the broken upper level
            else:
                signals[i] = 0.0
    
    return signals