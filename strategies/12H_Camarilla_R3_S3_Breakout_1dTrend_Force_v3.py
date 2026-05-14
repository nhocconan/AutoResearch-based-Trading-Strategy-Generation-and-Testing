# 12H_Camarilla_R3_S3_Breakout_1dTrend_Force_v3
# Hypothesis: Camarilla pivot levels on 12h identify key support/resistance. Breakout confirmed by 1d trend (EMA34) and volume surge. Works in bull markets by catching breakouts and in bear markets by catching sharp reversals at key levels with volume confirmation. Target: 12-37 trades/year.

#!/usr/bin/env python3
name = "12H_Camarilla_R3_S3_Breakout_1dTrend_Force_v3"
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
    
    # Calculate 12h period Camarilla levels (using previous 12h bar)
    R3 = np.zeros(n)
    S3 = np.zeros(n)
    for i in range(1, n):
        # Previous bar OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:
            R3[i] = prev_close + range_val * 1.1 / 2
            S3[i] = prev_close - range_val * 1.1 / 2
        else:
            R3[i] = prev_close
            S3[i] = prev_close
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA34 on daily close
    ema_34 = np.zeros_like(close_1d)
    ema_34[:] = np.nan
    alpha = 2 / (34 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_34[i] = close_1d[i]
        elif np.isnan(ema_34[i-1]):
            ema_34[i] = close_1d[i]
        else:
            ema_34[i] = alpha * close_1d[i] + (1 - alpha) * ema_34[i-1]
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    vol_ma_20[:] = np.nan
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Close > R3 + volume spike + 1d uptrend (close > EMA34)
            if (close[i] > R3[i] and vol_spike and close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S3 + volume spike + 1d downtrend (close < EMA34)
            elif (close[i] < S3[i] and vol_spike and close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close < S3 (reversal to opposite level)
            if close[i] < S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close > R3 (reversal to opposite level)
            if close[i] > R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals