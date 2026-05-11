#!/usr/bin/env python3
name = "6h_Keltner_Trend_MeanRev"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and mean reversion
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily 20-period EMA for trend
    daily_close = df_1d['close'].values
    daily_ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema20_aligned = align_htf_to_ltf(prices, df_1d, daily_ema20)
    
    # Daily 14-period ATR for Keltner channels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_prev = np.roll(daily_close, 1)
    daily_close_prev[0] = daily_close[0]
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - daily_close_prev)
    tr3 = np.abs(daily_low - daily_close_prev)
    daily_tr = np.maximum(tr1, np.maximum(tr2, tr3))
    daily_atr14 = pd.Series(daily_tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    daily_atr14_aligned = align_htf_to_ltf(prices, df_1d, daily_atr14)
    
    # Keltner channels from daily data
    keltner_upper = daily_ema20_aligned + (2.0 * daily_atr14_aligned)
    keltner_lower = daily_ema20_aligned - (2.0 * daily_atr14_aligned)
    
    # 6h RSI for mean reversion entry
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Volume filter
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price pulls back to lower Keltner in uptrend + RSI oversold + volume
            if (close[i] <= keltner_lower[i] and 
                close[i] > daily_ema20_aligned[i] and 
                rsi[i] < 30 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price rallies to upper Keltner in downtrend + RSI overbought + volume
            elif (close[i] >= keltner_upper[i] and 
                  close[i] < daily_ema20_aligned[i] and 
                  rsi[i] > 70 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above daily EMA20 or RSI overbought
            if close[i] >= daily_ema20_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below daily EMA20 or RSI oversold
            if close[i] <= daily_ema20_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals