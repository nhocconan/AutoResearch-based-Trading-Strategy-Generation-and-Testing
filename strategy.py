#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Donchian breakout captures momentum in trending markets (works in both bull/bear via trend filter)
# Long when price breaks above Donchian(20) high AND price > 12h EMA50 (uptrend)
# Short when price breaks below Donchian(20) low AND price < 12h EMA50 (downtrend)
# Volume confirmation (>1.3x 20-period average) ensures institutional participation
# ATR-based stoploss (2.5x ATR) and time-based exit (max 12 bars) control risk
# Target: 20-40 trades/year on 4h timeframe to minimize fee drag

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation (on 4h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    start_idx = 20  # Donchian and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_atr = atr[i]
        
        # Update bars since entry
        if position != 0:
            bars_since_entry += 1
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit conditions: Donchian break below low, stoploss, or max time
            if (curr_low <= lowest_low[i] or  # Donchian breakout exit
                curr_close <= entry_price - 2.5 * curr_atr or  # ATR stoploss
                bars_since_entry >= 12):  # Time-based exit (max 12 bars = 2 days)
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Donchian break above high, stoploss, or max time
            if (curr_high >= highest_high[i] or  # Donchian breakout exit
                curr_close >= entry_price + 2.5 * curr_atr or  # ATR stoploss
                bars_since_entry >= 12):  # Time-based exit (max 12 bars = 2 days)
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.3x 20-period average
            vol_confirm = curr_volume > 1.3 * curr_vol_ma
            
            # Long entry: price breaks above Donchian high in uptrend (price > 12h EMA50)
            if vol_confirm and curr_close > curr_ema50_12h:
                if curr_high > highest_high[i]:  # Break above Donchian high
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    bars_since_entry = 0
            # Short entry: price breaks below Donchian low in downtrend (price < 12h EMA50)
            elif vol_confirm and curr_close < curr_ema50_12h:
                if curr_low < lowest_low[i]:  # Break below Donchian low
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    bars_since_entry = 0
            else:
                signals[i] = 0.0
    
    return signals