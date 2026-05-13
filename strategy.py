#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (EMA50) and volume confirmation. 
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance. 
# Breakout above R3 with 1d EMA50 uptrend and volume > 1.5x 20-period average → long. 
# Breakdown below S3 with 1d EMA50 downtrend and volume confirmation → short. 
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 12h timeframe to avoid overtrading while capturing strong momentum moves in both bull and bear markets.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback for volume MA
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Need prior bar's high/low/close for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Camarilla pivot levels from prior bar
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Camarilla levels
        R3 = pc + (ph - pl) * 1.1 / 4
        S3 = pc - (ph - pl) * 1.1 / 4
        
        if position == 0:
            # LONG: Break above R3 with 1d EMA50 uptrend and volume confirmation
            if (close[i] > R3 and 
                close[i] > ema_50_1d_aligned[i] and  # price above 1d EMA50 (uptrend)
                volume[i] > 1.5 * vol_ma_20[i]):   # volume > 1.5x 20-period average
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with 1d EMA50 downtrend and volume confirmation
            elif (close[i] < S3 and 
                  close[i] < ema_50_1d_aligned[i] and  # price below 1d EMA50 (downtrend)
                  volume[i] > 1.5 * vol_ma_20[i]):   # volume > 1.5x 20-period average
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below prior bar's close (simple mean reversion exit)
            if close[i] < pc:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above prior bar's close
            if close[i] > pc:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals