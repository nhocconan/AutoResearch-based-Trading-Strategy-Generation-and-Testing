#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Reversal_1dTrend_Volume
# Hypothesis: In 12-hour timeframe, trade reversals at Camarilla pivot levels (S3/R3) with
# 1-day trend filter and volume confirmation. In uptrend (price > 1-day EMA34), look for long
# setups at S3 support; in downtrend (price < 1-day EMA34), look for short setups at R3 resistance.
# Uses volume spike (>1.5x average) to confirm institutional interest at pivot levels.
# Target: 12-30 trades per year (~48-120 over 4 years) with position size 0.25.

name = "12h_Camarilla_Pivot_Reversal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's range
    # For each 12h bar, we use the previous day's high, low, close
    # We need to align the previous day's data to current 12h bars
    
    # Extract previous day's OHLC for each bar
    prev_day_high = np.roll(high, 2)  # 2 periods back = previous day (assuming 2x 12h per day)
    prev_day_low = np.roll(low, 2)
    prev_day_close = np.roll(close, 2)
    
    # Handle first two bars (no previous day data)
    prev_day_high[:2] = np.nan
    prev_day_low[:2] = np.nan
    prev_day_close[:2] = np.nan
    
    # Calculate Camarilla levels
    range_prev = prev_day_high - prev_day_low
    camarilla_s3 = prev_day_close - (1.1 * range_prev / 6)
    camarilla_r3 = prev_day_close + (1.1 * range_prev / 6)
    
    # Volume ratio: current volume / 30-period average volume
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need 40 periods for EMA34 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(prev_day_high[i]) or np.isnan(prev_day_low[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA34
        uptrend_regime = close[i] > ema_34_1d_aligned[i]
        downtrend_regime = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long setup: price at S3 support in uptrend with volume
            long_setup = (abs(close[i] - camarilla_s3[i]) < 0.001 * camarilla_s3[i]) and uptrend_regime and volume_confirm
            # Short setup: price at R3 resistance in downtrend with volume
            short_setup = (abs(close[i] - camarilla_r3[i]) < 0.001 * camarilla_r3[i]) and downtrend_regime and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches R3 level or trend changes to downtrend
            if (close[i] >= camarilla_r3[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches S3 level or trend changes to uptrend
            if (close[i] <= camarilla_s3[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals