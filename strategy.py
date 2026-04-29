#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In bull markets (price > 1d EMA50), we look for Bull Power expansion with volume
# In bear markets (price < 1d EMA50), we look for Bear Power expansion with volume
# Uses discrete position sizing (0.25) to minimize fee churn and target ~50-150 trades over 4 years
# Works in both bull and bear via 1d EMA50 trend filter - trades only in alignment with higher timeframe momentum

name = "6h_ElderRay_BullBearPower_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_ema13 = ema_13[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits: flatten when price crosses 1d EMA50 (trend change)
        if position == 1 and curr_close < curr_ema50_1d:
            signals[i] = 0.0
            position = 0
        elif position == -1 and curr_close > curr_ema50_1d:
            signals[i] = 0.0
            position = 0
        else:
            # Volume confirmation: current volume > 1.8x 20-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Look for new entries only when flat
            if position == 0:
                # Long when: price > 1d EMA50 (bullish regime) AND Bull Power expanding AND volume confirmation
                if curr_close > curr_ema50_1d and curr_bull_power > 0 and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short when: price < 1d EMA50 (bearish regime) AND Bear Power expanding AND volume confirmation
                elif curr_close < curr_ema50_1d and curr_bear_power < 0 and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals