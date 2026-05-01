#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 12h EMA50 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions. Extreme readings (<-90 or >-10) combined with
# 12h EMA50 trend filter and volume spike (>2x 20-bar MA) provide high-probability mean reversals in ranging
# markets and continuation in trending markets. Works in bull markets via buying dips in uptrend, in bear markets
# via selling rallies in downtrend. Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).

name = "6h_WilliamsR_Extreme_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50
    close_12h = pd.Series(df_12h['close'])
    ema_50_12h = close_12h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Williams %R(14) on 6h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(14, 20, 50)  # Williams %R(14), volume MA(20), EMA50
    
    for i in range(start_idx, n):
        if np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Williams %R extreme levels
        wr_oversold = williams_r[i] < -90
        wr_overbought = williams_r[i] > -10
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold, price above 12h EMA50, volume spike
            if wr_oversold and curr_close > ema_50_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought, price below 12h EMA50, volume spike
            elif wr_overbought and curr_close < ema_50_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R overbought or price below 12h EMA50
            if wr_overbought or curr_close < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R oversold or price above 12h EMA50
            if wr_oversold or curr_close > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals