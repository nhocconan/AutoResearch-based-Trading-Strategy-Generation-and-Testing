#!/usr/bin/env python3
# 1D_Camarilla_R1S1_Breakout_1wTrend_Volume
# Hypothesis: Use Camarilla pivot levels (R1/S1) on 1d combined with 1w trend filter and volume confirmation.
# In bull markets: buy near S1 with 1w uptrend, sell at R1. In bear markets: sell near R1 with 1w downtrend, buy at S1.
# The Camarilla levels provide precise intraday support/resistance, while 1w trend ensures higher timeframe alignment.
# Volume confirmation filters out false breakouts. Target: 15-25 trades/year to minimize fee drag.

name = "1D_Camarilla_R1S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Calculate Camarilla levels for current day using previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's data, so we shift by 1
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First value has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    cam_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    cam_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # 1-week trend filter: EMA(34) on 1w close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5 * 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(cam_r1[i]) or np.isnan(cam_s1[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above S1 with 1w uptrend and volume confirmation
            if close[i] > cam_s1[i] and close[i-1] <= cam_s1[i-1] and \
               close[i] > ema_34_1w_aligned[i] and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with 1w downtrend and volume confirmation
            elif close[i] < cam_r1[i] and close[i-1] >= cam_r1[i-1] and \
                 close[i] < ema_34_1w_aligned[i] and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below S1 OR 1w trend turns down
            if close[i] < cam_s1[i] and close[i-1] >= cam_s1[i-1] or \
               close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above R1 OR 1w trend turns up
            if close[i] > cam_r1[i] and close[i-1] <= cam_r1[i-1] or \
               close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals