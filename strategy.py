#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R for mean reversion in extreme zones with volume confirmation.
# Long when 1d Williams %R < -80 (oversold) + price > 4h VWAP + volume > 1.5x 20-period average.
# Short when 1d Williams %R > -20 (overbought) + price < 4h VWAP + volume > 1.5x 20-period average.
# Exit when Williams %R returns to neutral zone (-80 to -20) or volume drops below average.
# Uses discrete position size 0.25. Williams %R identifies exhaustion points, VWAP provides dynamic support/resistance,
# volume confirmation ensures institutional participation. Target: 80-180 total trades over 4 years (20-45/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation (14-period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 13 or np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i]) or (highest_high_14[i] - lowest_low_14[i]) == 0:
            williams_r[i] = -50.0  # neutral
        else:
            williams_r[i] = (highest_high_14[i] - close_1d[i]) / (highest_high_14[i] - lowest_low_14[i]) * -100
    
    # Align 1d Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 4h data once before loop for VWAP and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Typical price for VWAP
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    vp = typical_price_4h * volume_4h
    
    # Cumulative VWAP (reset periodically)
    cum_vp = np.cumsum(vp)
    cum_vol = np.cumsum(volume_4h)
    vwap = np.divide(cum_vp, cum_vol, out=np.zeros_like(cum_vp), where=cum_vol!=0)
    
    # Align 4h VWAP to 4h timeframe (no alignment needed as same timeframe)
    vwap_aligned = vwap  # already aligned
    
    # Volume moving average (20-period) on 4h
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = vol_ma_20_4h  # already aligned
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        vwap_val = vwap_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R returns to neutral zone or price drops below VWAP
            if wr > -80 or price < vwap_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R returns to neutral zone or price rises above VWAP
            if wr < -20 or price > vwap_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma_val
            
            # Price filter: price must be on correct side of VWAP
            price_filter_long = price > vwap_val
            price_filter_short = price < vwap_val
            
            # LONG: Williams %R oversold (< -80), price > VWAP, volume confirmation
            if wr < -80 and price_filter_long and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R overbought (> -20), price < VWAP, volume confirmation
            elif wr > -20 and price_filter_short and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dWilliamsR_VWAP_VolumeConfirmation_V1"
timeframe = "4h"
leverage = 1.0