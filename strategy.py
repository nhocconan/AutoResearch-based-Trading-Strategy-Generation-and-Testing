#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 (bullish breakout) AND price > 1w EMA34 AND volume > 1.5 * 20-period average volume.
# Short when price breaks below S3 (bearish breakout) AND price < 1w EMA34 AND volume > 1.5 * 20-period average volume.
# Exits when price returns to the Camarilla pivot point (mean reversion to equilibrium).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing breakouts in trending markets with volume confirmation.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_VolumeConfirm_v1"
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
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period average volume for volume confirmation
    if len(volume) < 20:
        vol_ma = np.full(n, np.nan)
    else:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after volume MA warmup
        # Need previous day's OHLC for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate Camarilla levels
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        r3 = pivot + (range_val * 1.1 / 4)
        s3 = pivot - (range_val * 1.1 / 4)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND price > 1w EMA34 AND volume > 1.5 * avg volume
            if (close[i] > r3 and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND price < 1w EMA34 AND volume > 1.5 * avg volume
            elif (close[i] < s3 and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot (mean reversion)
            if close[i] <= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot (mean reversion)
            if close[i] >= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals