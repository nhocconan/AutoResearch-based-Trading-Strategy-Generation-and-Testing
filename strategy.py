#!/usr/bin/env python3
"""
4h_1d_RSI_MeanReversion_With_Trend_Filter
Hypothesis: 4h timeframe with RSI mean-reversion (RSI<30 long, RSI>70 short) filtered by 1d trend (price above/below 200 EMA) and volume confirmation. 
Trades only in direction of higher timeframe trend to avoid counter-trend whipsaws. 
Designed for low trade frequency (<400 total) and works in both bull/bear markets by following the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 200 EMA on 1d close
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # RSI on 4h close
    close = prices['close'].values
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if indicators not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: RSI oversold + price above 1d EMA200 + volume
            if rsi[i] < 30 and price > ema_200_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + price below 1d EMA200 + volume
            elif rsi[i] > 70 and price < ema_200_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or trend fails
            if rsi[i] > 50 or price < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or trend fails
            if rsi[i] < 50 or price > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_RSI_MeanReversion_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0