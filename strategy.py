#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA200 trend filter and volume confirmation (>1.5x 20-period average).
# Long when price breaks above R3 AND close > 1w EMA200 (bullish trend) AND volume > 1.5x MA20.
# Short when price breaks below S3 AND close < 1w EMA200 (bearish trend) AND volume > 1.5x MA20.
# Exit when price crosses the 1w EMA200 in opposite direction.
# Uses 1w HTF EMA200 for strong trend filter to avoid whipsaws and reduce trade frequency.
# Volume confirmation reduces false breakouts.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

name = "1d_Camarilla_R3S3_Breakout_1wEMA200_VolumeConfirm_v1"
timeframe = "1d"
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
    
    # --- 1d Indicators (LTF) ---
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(200) - strong trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for today (using previous day's OHLC)
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_ = prev_high - prev_low
            
            # Camarilla levels
            R3 = prev_close + range_ * 1.1 / 2
            S3 = prev_close - range_ * 1.1 / 2
        else:
            R3 = np.nan
            S3 = np.nan
        
        if position == 0:
            # LONG: Price breaks above R3 AND close > 1w EMA200 (bullish trend) AND volume confirm
            if (not np.isnan(R3) and 
                close[i] > R3 and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND close < 1w EMA200 (bearish trend) AND volume confirm
            elif (not np.isnan(S3) and 
                  close[i] < S3 and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1w EMA200 (trend change)
            if close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1w EMA200 (trend change)
            if close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals