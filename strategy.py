#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter, volume spike (>2.0x average), and chop regime filter (CHOP < 61.8)
# Uses 12h HTF for trend alignment as specified in experiment instructions. Volume threshold increased to 2.0x to reduce trades.
# Discrete position sizing 0.25 to minimize fee churn. Target: 75-200 trades over 4 years.

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Volume_Chop"
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
    
    # Calculate Camarilla levels for R3 and S3 (using prior day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Prior day high
    prev_low = df_1d['low'].shift(1).values    # Prior day low
    prev_close = df_1d['close'].shift(1).values # Prior day close
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (they update daily)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 12h EMA50 for trend filter (using HTF as specified)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 2.0x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index regime filter (avoid ranging markets)
    # CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((highest_high - lowest_low) / (atr_safe * np.sqrt(atr_period))) / np.log10(atr_period)
    chop_regime = chop < 61.8  # True when trending (CHOP < 61.8), False when ranging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3 AND price > 12h EMA50 AND volume spike AND trending regime
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND price < 12h EMA50 AND volume spike AND trending regime
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below S3 OR price < 12h EMA50
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above R3 OR price > 12h EMA50
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals