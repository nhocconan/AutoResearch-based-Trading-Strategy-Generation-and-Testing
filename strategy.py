#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA trend filter and volume confirmation.
# Long: Close breaks above Camarilla R3 AND 4h EMA50 > EMA200 (uptrend) AND volume > 1.5x 20-period MA
# Short: Close breaks below Camarilla S3 AND 4h EMA50 < EMA200 (downtrend) AND volume > 1.5x 20-period MA
# Exit: Opposite Camarilla breakout or EMA crossover reversal or volume drops.
# Discrete sizing 0.20. Target: 60-150 total trades over 4 years (15-37/year).
# Camarilla provides precise intraday support/resistance; 4h EMA filter ensures alignment with higher timeframe trend;
# volume confirmation reduces false breakouts. Works in bull via long signals and bear via short signals when aligned with trend.

name = "1h_Camarilla_R3S3_4hEMA_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 and EMA200
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200)
    
    # Calculate Camarilla levels (using previous day's OHLC)
    # For 1h timeframe, we use daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume regime: current 1h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_50_val = ema_50_aligned[i]
        ema_200_val = ema_200_aligned[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = ema_50_val > ema_200_val
        is_downtrend = ema_50_val < ema_200_val
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Camarilla R3 AND uptrend AND volume spike
            if close_val > camarilla_r3_val and is_uptrend and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: Close breaks below Camarilla S3 AND downtrend AND volume spike
            elif close_val < camarilla_s3_val and is_downtrend and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Camarilla S3 OR trend reverses (EMA50 < EMA200) OR volume drops
            if close_val < camarilla_s3_val or not is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close breaks above Camarilla R3 OR trend reverses (EMA50 > EMA200) OR volume drops
            if close_val > camarilla_r3_val or not is_downtrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals