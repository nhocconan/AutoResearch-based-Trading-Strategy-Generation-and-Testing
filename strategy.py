#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1w trend filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0 (bullish divergence), price > 1w EMA50, and volume > 1.5x 20 EMA.
# Short when Bear Power < 0, Bull Power < 0 (bearish divergence), price < 1w EMA50, and volume > 1.5x 20 EMA.
# Uses weekly trend for direction and Elder Ray for momentum exhaustion signals.
# Designed for moderate trade frequency (target: 20-40/year) to balance opportunity and cost.
# Works in bull markets via bullish divergence and in bear via bearish divergence with trend filter.
name = "6h_ElderRay_1wTrend_Volume"
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
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d volume > 1.5x 20 EMA
    vol_ema_20_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm_1d = np.where(vol_ema_20_1d > 0, df_1d['volume'].values / vol_ema_20_1d, 1.0) > 1.5
    vol_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm_1d)
    
    # Calculate Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) on 6h
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA13 and aligned arrays
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_confirm_1d_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price > 1w EMA50, volume confirmation
            long_cond = (bull_power[i] > 0) and (bear_power[i] < 0) and (close[i] > ema_50_1w_aligned[i]) and vol_confirm_1d_aligned[i]
            # Short: Bear Power < 0, Bull Power < 0, price < 1w EMA50, volume confirmation
            short_cond = (bear_power[i] < 0) and (bull_power[i] < 0) and (close[i] < ema_50_1w_aligned[i]) and vol_confirm_1d_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bear Power becomes positive (bullish momentum fails) or price < 1w EMA50
            if (bear_power[i] >= 0) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bull Power becomes negative (bearish momentum fails) or price > 1w EMA50
            if (bull_power[i] <= 0) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals