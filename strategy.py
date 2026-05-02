#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Uses 4h EMA34 for trend direction (bull/bear filter) and 1h Camarilla levels for precise entries
# Entry logic: Long when price breaks above 1h Camarilla R3 with volume spike and price > 4h EMA34 (uptrend)
#              Short when price breaks below 1h Camarilla S3 with volume spike and price < 4h EMA34 (downtrend)
# Exit logic: Exit when price crosses the 4h EMA34 (trend reversal) or opposite Camarilla level (R4/S4)
# Works in both bull and bear markets by trading with the 4h trend
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Discrete sizing 0.20 minimizes fee churn while maintaining profit potential

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA34 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h Camarilla levels (based on previous day's OHLC)
    # Need to resample to daily to get previous day's OHLC for Camarilla calculation
    from mtf_data import get_htf_data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1h bar using previous completed day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #                 S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.125 * (high_1d - low_1d)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 1h timeframe (use previous completed day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above 1h Camarilla R3 AND price > 4h EMA34 (uptrend) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: Break below 1h Camarilla S3 AND price < 4h EMA34 (downtrend) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 4h EMA34 (trend change) OR break below 1h Camarilla S4 (strong reversal)
            if (close[i] < ema_34_4h_aligned[i] or 
                close[i] < camarilla_s4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Close above 4h EMA34 (trend change) OR break above 1h Camarilla R4 (strong reversal)
            if (close[i] > ema_34_4h_aligned[i] or 
                close[i] > camarilla_r4_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals