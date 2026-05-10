#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike
# Hypothesis: Long when price breaks above Camarilla R3 level with volume > 2x average and price above 1d EMA34.
# Short when price breaks below Camarilla S3 level with volume > 2x average and price below 1d EMA34.
# Exit when price crosses back below R3 (long) or above S3 (short).
# Uses Camarilla pivot levels for institutional support/resistance, effective in both bull and bear markets.
# Designed for 20-50 trades/year to avoid fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Calculate Camarilla levels from previous day
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    for i in range(1, n):
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_ = prev_high - prev_low
        camarilla_R3[i] = prev_close + range_ * 1.1 / 4
        camarilla_S3[i] = prev_close - range_ * 1.1 / 4
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and price above 1d EMA34
            if close[i] > camarilla_R3[i] and close[i-1] <= camarilla_R3[i-1] and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and price below 1d EMA34
            elif close[i] < camarilla_S3[i] and close[i-1] >= camarilla_S3[i-1] and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses back below R3
            if close[i] < camarilla_R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses back above S3
            if close[i] > camarilla_S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals