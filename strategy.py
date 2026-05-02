#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels provide institutional support/resistance. Breakout of R3/S3
# with volume confirmation and 1d EMA34 trend filter captures strong momentum moves.
# Works in both bull and bear markets by aligning with higher-timeframe trend.
# Target: 75-200 trades over 4 years (19-50/year) on 4h.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Need at least 2 days of data (current day + previous day)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].iloc[-2]  # Second to last day
    prev_low = df_1d['low'].iloc[-2]
    prev_close = df_1d['close'].iloc[-2]
    
    # Camarilla levels
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + (camarilla_range * 1.1 / 4)
    camarilla_s3 = prev_close - (camarilla_range * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (constant until new 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), camarilla_r3))
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), camarilla_s3))
    
    # Volume confirmation: 2.0x 20-period average
    if len(volume) < 20:
        return np.zeros(n)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA and volume)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 with volume spike AND price > 1d EMA34 (bullish trend)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S3 with volume spike AND price < 1d EMA34 (bearish trend)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Camarilla pivot (central) OR below 1d EMA34 (trend change)
            camarilla_pivot = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2.0
            if close[i] < camarilla_pivot or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla pivot (central) OR above 1d EMA34 (trend change)
            camarilla_pivot = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2.0
            if close[i] > camarilla_pivot or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals