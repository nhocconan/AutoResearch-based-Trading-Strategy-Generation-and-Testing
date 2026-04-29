#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation + ATR-based trailing stoploss
# Long when price breaks above 20-period Donchian high AND price > 12h EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below 20-period Donchian low AND price < 12h EMA50 AND volume > 1.5x 20-bar avg
# Exit when price retraces 50% of the ATR from extreme (trailing stop) or reverses to opposite Donchian band
# Uses discrete position sizing (0.25) to balance capture and drawdown. Target: 25-50 trades/year on 4h timeframe.
# Donchian channels provide structural breakouts, 12h EMA50 filters counter-trend moves in bear markets,
# volume confirmation reduces false breakouts, ATR stop manages risk without look-ahead.

name = "4h_Donchian20_12hEMA50_VolumeSpike_ATRTrail_v1"
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
    
    # Get 12h data for EMA50 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_entry_price = 0.0
    short_entry_price = 0.0
    long_highest_since_entry = 0.0
    short_lowest_since_entry = 0.0
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50 = ema_50_12h_aligned[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        curr_atr = atr[i]
        
        # Handle position exits and trailing stops
        if position == 1:  # Long position
            # Update highest high since entry
            if curr_high > long_highest_since_entry:
                long_highest_since_entry = curr_high
            
            # Exit conditions:
            # 1. Trailing stop: price drops 2.0*ATR from highest since entry
            # 2. Reversal: price breaks below Donchian lower band
            if curr_close < long_highest_since_entry - 2.0 * curr_atr or curr_close < curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < short_lowest_since_entry:
                short_lowest_since_entry = curr_low
            
            # Exit conditions:
            # 1. Trailing stop: price rises 2.0*ATR from lowest since entry
            # 2. Reversal: price breaks above Donchian upper band
            if curr_close > short_lowest_since_entry + 2.0 * curr_atr or curr_close > curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper band AND price > 12h EMA50 AND volume confirmation
            if curr_close > curr_upper and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
                long_entry_price = curr_close
                long_highest_since_entry = curr_high
            # Short when price breaks below Donchian lower band AND price < 12h EMA50 AND volume confirmation
            elif curr_close < curr_lower and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
                short_entry_price = curr_close
                short_lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals