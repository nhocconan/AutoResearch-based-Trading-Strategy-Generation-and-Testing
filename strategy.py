#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R3 with 12h uptrend and volume spike.
# Short when price breaks below S3 with 12h downtrend and volume spike.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 50-150 total trades over 4 years.
# Camarilla levels provide intraday support/resistance, 12h EMA50 ensures higher timeframe alignment,
# Volume spike confirms institutional interest. Works in both bull and bear markets by only trading
# with the 12h trend, avoiding counter-trend whipsaws. Designed for 6h timeframe to minimize fee drag.

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla pivot calculation
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Calculate 6h Camarilla levels (using previous bar's H/L/C)
    prev_high = np.roll(df_6h['high'].values, 1)
    prev_low = np.roll(df_6h['low'].values, 1)
    prev_close = np.roll(df_6h['close'].values, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R3, S3, R4, S4
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    r4 = prev_close + camarilla_range * 1.1 / 2
    s4 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        
        if position == 0:
            # Long: price breaks above R3 AND 12h uptrend AND volume spike
            if close_val > r3_aligned[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND 12h downtrend AND volume spike
            elif close_val < s3_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (reversal) OR breaks above R4 (exhaustion)
            if close_val < s3_aligned[i] or close_val > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (reversal) OR breaks below S4 (exhaustion)
            if close_val > r3_aligned[i] or close_val < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals