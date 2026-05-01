#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme with 1d EMA(34) trend filter and volume confirmation (>1.5x 20-bar MA)
# Williams %R identifies overbought/oversold conditions. Long when %R < -80 (oversold) with uptrend,
# short when %R > -20 (overbought) with downtrend. Works in both bull (buy dips in uptrend) and 
# bear (sell rallies in downtrend). Volume confirmation ensures breakout validity. Target: 50-150 total trades.

name = "12h_WilliamsR_Extreme_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(34) on 1d close
    daily_ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA to 12h timeframe
    daily_ema_34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema_34)
    
    # Williams %R calculation (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(14, 20, 34)  # Need 14 for Williams %R, 20 for volume MA, 34 for EMA
    
    for i in range(start_idx, n):
        if np.isnan(daily_ema_34_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80), price above daily EMA, and volume confirmation
            if curr_williams_r < -80 and curr_close > daily_ema_34_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price below daily EMA, and volume confirmation
            elif curr_williams_r > -20 and curr_close < daily_ema_34_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R rising above -50 or price below daily EMA
            if curr_williams_r > -50 or curr_close < daily_ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R falling below -50 or price above daily EMA
            if curr_williams_r < -50 or curr_close > daily_ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals