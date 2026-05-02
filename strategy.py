#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Camarilla R3/S3 levels act as strong support/resistance from 1d price action
# Breakout above R3 or below S3 with volume confirmation captures institutional moves
# 1d EMA34 filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# Works in bull via breakout continuation and bear via avoidance of false breakouts
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 day for Camarilla calculation
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #                 S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for 1d
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA(34) for trend filter
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla and EMA)
    start_idx = 35  # Need 34 for EMA plus 1 for shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 OR price < 1d EMA34
            if (close[i] < camarilla_s3_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 OR price > 1d EMA34
            if (close[i] > camarilla_r3_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals