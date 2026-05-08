#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI_Divergence_Trend_Confirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # RSI divergence detection
    rsi_peak = np.zeros(n, dtype=bool)
    rsi_trough = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        # RSI peak: higher high in price, lower high in RSI
        if (high[i] > high[i-1] and high[i] > high[i+1] and
            rsi[i] > rsi[i-1] and rsi[i] > rsi[i+1] and
            rsi[i] < rsi[i-2]):  # lower RSI high
            rsi_peak[i] = True
        # RSI trough: lower low in price, higher low in RSI
        if (low[i] < low[i-1] and low[i] < low[i+1] and
            rsi[i] < rsi[i-1] and rsi[i] < rsi[i+1] and
            rsi[i] > rsi[i-2]):  # higher RSI low
            rsi_trough[i] = True
    
    # Align divergence signals
    rsi_peak_aligned = align_htf_to_ltf(prices, df_1d, rsi_peak.astype(float))
    rsi_trough_aligned = align_htf_to_ltf(prices, df_1d, rsi_trough.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi_peak_aligned[i]) or 
            np.isnan(rsi_trough_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI bullish divergence + uptrend (price > 1d EMA34) + volume
            long_cond = rsi_trough_aligned[i] > 0.5 and \
                        (close[i] > ema_34_1d_aligned[i]) and \
                        volume_filter[i]
            # Short: RSI bearish divergence + downtrend (price < 1d EMA34) + volume
            short_cond = rsi_peak_aligned[i] > 0.5 and \
                         (close[i] < ema_34_1d_aligned[i]) and \
                         volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI bearish divergence or price below EMA
            if (rsi_peak_aligned[i] > 0.5) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI bullish divergence or price above EMA
            if (rsi_trough_aligned[i] > 0.5) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals