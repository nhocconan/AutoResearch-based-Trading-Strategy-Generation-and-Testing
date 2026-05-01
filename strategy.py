#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d EMA(34) trend + volume confirmation
# Williams %R identifies overbought/oversold conditions. In strong trends (1d EMA alignment),
# extreme readings can signal continuation rather than reversal. Volume confirms conviction.
# Works in bull markets: buy Williams %R < -90 (oversold) with uptrend + volume.
# Works in bear markets: sell Williams %R > -10 (overbought) with downtrend + volume.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(34) on 1d close
    daily_ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA to 6h timeframe
    daily_ema_34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema_34)
    
    # Williams %R on 6h data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        if np.isnan(daily_ema_34_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -90 (extreme oversold), price above daily EMA, volume confirmation
            if curr_williams_r < -90 and curr_close > daily_ema_34_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -10 (extreme overbought), price below daily EMA, volume confirmation
            elif curr_williams_r > -10 and curr_close < daily_ema_34_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Williams %R > -50 (return to neutral) or price below daily EMA
            if curr_williams_r > -50 or curr_close < daily_ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Williams %R < -50 (return to neutral) or price above daily EMA
            if curr_williams_r < -50 or curr_close > daily_ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals