#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation (>2.0x 20-period average).
# Long when price breaks above R3 AND close > 1d EMA50 AND volume > 2.0x MA20.
# Short when price breaks below S3 AND close < 1d EMA50 AND volume > 2.0x MA20.
# Exit when price crosses the 1d EMA50 in opposite direction.
# Uses 1d HTF for primary trend to reduce noise and overtrading. Higher volume threshold reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
# Camarilla R3/S3 levels provide good support/resistance with balanced trade frequency.

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # Volume confirmation: > 2.0x 20-period average (high threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for today (using previous bar's OHLC)
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_ = prev_high - prev_low
            
            # Camarilla levels (R3/S3 = standard levels)
            R3 = prev_close + range_ * 1.1/4
            S3 = prev_close - range_ * 1.1/4
        else:
            R3 = np.nan
            S3 = np.nan
        
        if position == 0:
            # LONG: Price breaks above R3 AND close > 1d EMA50 (bullish trend) AND volume confirm
            if (not np.isnan(R3) and 
                close[i] > R3 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND close < 1d EMA50 (bearish trend) AND volume confirm
            elif (not np.isnan(S3) and 
                  close[i] < S3 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1d EMA50 (trend change)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1d EMA50 (trend change)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals