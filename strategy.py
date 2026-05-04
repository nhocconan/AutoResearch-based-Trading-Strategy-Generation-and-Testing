#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels provide precise intraday support/resistance derived from prior 1d range.
# Breakout of R3 (resistance 3) or S3 (support 3) with 1w EMA50 trend alignment captures strong momentum.
# Volume confirmation (>1.5x 50-period EMA) filters false breakouts. Discrete sizing 0.25 limits risk.
# Works in bull/bear: 1w EMA50 trend filter prevents counter-trend entries. Target: 50-150 trades over 4 years.

name = "12h_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values  # needed for Camarilla calculation
    
    # Get 1d data for Camarilla pivot calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 50-period EMA of volume on 12h timeframe
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup for EMA50
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Need prior 1d OHLC for Camarilla (must be completed 1d bar)
        # Find index of completed 1d bar prior to current 12h bar
        # Since we're on 12h timeframe, we can use prior 1d bar's close
        # Get prior completed 1d bar index: we need df_1d index that corresponds to completed day
        # align_htf_to_ltf already handles this - we'll get 1d values aligned to 12h
        # For Camarilla, we need the prior 1day's OHLC, so we shift the 1d data by 1
        
        # Get aligned 1d OHLC (already aligned to 12h timeframe via align_htf_to_ltf)
        # We'll compute Camarilla levels using prior 1d bar's OHLC
        if i < 1:  # need at least one prior 1d bar
            continue
            
        # Get prior completed 1d bar's OHLC aligned to current 12h bar
        # We need to shift the 1d data by 1 bar to get prior day's values
        # Since we can't shift inside loop efficiently, we'll precompute
        
        # Precompute Camarilla levels outside loop for efficiency
        pass  # will implement after precomputation
    
    # Precompute indicators before loop for efficiency
    # Get 1d data for Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate prior 1d bar's OHLC for Camarilla (shift by 1 to get completed prior day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get prior completed 1d bar's values (lookback 1)
    prior_high_1d = np.roll(high_1d, 1)
    prior_low_1d = np.roll(low_1d, 1)
    prior_close_1d = np.roll(close_1d, 1)
    # Set first value to NaN since no prior day
    prior_high_1d[0] = np.nan
    prior_low_1d[0] = np.nan
    prior_close_1d[0] = np.nan
    
    # Calculate Camarilla levels for prior 1d bar
    # Camarilla: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close)/3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    prior_range = prior_high_1d - prior_low_1d
    camarilla_r3 = prior_close_1d + (prior_range * 1.1 / 4)
    camarilla_s3 = prior_close_1d - (prior_range * 1.1 / 4)
    camarilla_r4 = prior_close_1d + (prior_range * 1.1 / 2)
    camarilla_s4 = prior_close_1d - (prior_range * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (prior 1d bar's levels are valid for entire next day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 50-period EMA of volume on 12h timeframe
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ema_50[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 50-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_50[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla R4 OR trend changes OR volume drops
            if (close[i] < camarilla_r4_aligned[i] or 
                close[i] < ema50_1w_aligned[i] or 
                volume[i] < vol_ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla S4 OR trend changes OR volume drops
            if (close[i] > camarilla_s4_aligned[i] or 
                close[i] > ema50_1w_aligned[i] or 
                volume[i] < vol_ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals