#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend filter (longer-term trend) and 6h Camarilla R3/S3 levels for breakout signals
# Entry logic: Long when price breaks above 6h Camarilla R3 with volume spike and price > 1d EMA34
#              Short when price breaks below 6h Camarilla S3 with volume spike and price < 1d EMA34
# Exit logic: Exit when price crosses the 1d EMA34 (trend reversal) or opposite Camarilla level (R4/S4)
# Works in both bull and bear markets by trading with the 1d trend
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Camarilla levels (based on previous 6h bar's OHLC)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    open_6h = df_6h['open'].values
    
    # Camarilla levels: based on previous 6h bar's range
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # R4 = close + 1.5*(high-low)/2, S4 = close - 1.5*(high-low)/2
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    prev_open = np.roll(open_6h, 1)
    
    # Set first values to NaN (no previous bar)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    prev_open[0] = np.nan
    
    # Calculate Camarilla levels for previous 6h bar
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    camarilla_r4 = prev_close + 1.5 * (prev_high - prev_low) / 2
    camarilla_s4 = prev_close - 1.5 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 6h timeframe (use previous completed 6h bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s4)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above 6h Camarilla R3 AND price > 1d EMA34 (uptrend) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below 6h Camarilla S3 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1d EMA34 (trend change) OR break below 6h Camarilla S4 (reversal)
            if (close[i] < ema_34_1d_aligned[i] or 
                close[i] < camarilla_s4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1d EMA34 (trend change) OR break above 6h Camarilla R4 (reversal)
            if (close[i] > ema_34_1d_aligned[i] or 
                close[i] > camarilla_r4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals