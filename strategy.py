#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Keltner Channel breakout with volume confirmation and 1w RSI trend filter.
# Long when price breaks above upper Keltner Channel (EMA + 2*ATR) with volume surge and 1w RSI > 50.
# Short when price breaks below lower Keltner Channel (EMA - 2*ATR) with volume surge and 1w RSI < 50.
# Uses 1d ATR and EMA for Keltner Channel, 1w RSI for trend filter.
# Designed for low trade frequency (15-25/year) to avoid fee drag. Keltner Channels adapt to volatility,
# working in both trending and ranging markets. Volume confirmation ensures breakout strength.

name = "4h_1dKeltner_1wRSI_Volume"
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
    
    # Get 1d data for Keltner Channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA (20-period) for Keltner Channel middle line
    ema_20 = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ATR (14-period) for Keltner Channel width
    tr1 = np.abs(df_1d['high'].values[1:] - df_1d['low'].values[:-1])
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Keltner Channel bands
    kc_upper = ema_20 + 2.0 * atr_14
    kc_lower = ema_20 - 2.0 * atr_14
    
    # Get 1w data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w RSI (14-period)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 4h timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower)
    
    # Align 1w RSI to 4h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: 4h volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kc_upper_aligned[i]) or 
            np.isnan(kc_lower_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper KC + volume surge + 1w RSI > 50
            if close[i] > kc_upper_aligned[i] and vol_spike[i] and rsi_1w_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower KC + volume surge + 1w RSI < 50
            elif close[i] < kc_lower_aligned[i] and vol_spike[i] and rsi_1w_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below middle line (EMA)
            if close[i] < ema_20_aligned[i] if 'ema_20_aligned' in locals() else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above middle line (EMA)
            if close[i] > ema_20_aligned[i] if 'ema_20_aligned' in locals() else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    # Fix: align EMA_20 for exit condition
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Re-run loop with proper EMA alignment (simplified - in practice we'd compute this earlier)
    # For brevity and correctness, we'll recompute signals with proper alignment
    
    # Recompute with proper EMA alignment
    signals = np.zeros(n)
    position = 0
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kc_upper_aligned[i]) or 
            np.isnan(kc_lower_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper KC + volume surge + 1w RSI > 50
            if close[i] > kc_upper_aligned[i] and vol_spike[i] and rsi_1w_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower KC + volume surge + 1w RSI < 50
            elif close[i] < kc_lower_aligned[i] and vol_spike[i] and rsi_1w_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below middle line (EMA)
            if close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above middle line (EMA)
            if close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals