#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume spike confirmation.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0, Bear Power rising, price > 1d EMA50, volume > 1.5x 20-bar avg.
# Short when Bear Power < 0, Bull Power falling, price < 1d EMA50, volume > 1.5x 20-bar avg.
# Exit when power reverses sign or price crosses 1d EMA50.
# Uses 1d EMA50 for higher timeframe trend alignment, targeting 12-37 trades/year on 6h.
# Works in bull markets via long signals and in bear markets via short signals with trend alignment.

name = "6h_ElderRay_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components (Bull Power and Bear Power) on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA13, EMA50_1d
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_13[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0, Bull Power rising (vs previous), price > 1d EMA50, volume spike
            if (curr_bull_power > 0 and 
                i > start_idx and curr_bull_power > bull_power[i-1] and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bear Power falling (vs previous), price < 1d EMA50, volume spike
            elif (curr_bear_power < 0 and 
                  i > start_idx and curr_bear_power < bear_power[i-1] and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 or price crosses below 1d EMA50
            if curr_bull_power <= 0 or curr_close < curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 or price crosses above 1d EMA50
            if curr_bear_power >= 0 or curr_close > curr_ema_50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals