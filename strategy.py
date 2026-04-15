#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily Bollinger Bands (20, 2) for mean reversion zones
    sma_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Volume filter: current volume > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            continue
        
        # Long conditions:
        # 1. Price below daily EMA50 (bearish trend)
        # 2. Price touches or breaks below daily lower Bollinger Band (oversold)
        # 3. Volume confirmation
        # 4. ATR filter: volatility not too low
        if (close[i] < ema_50_1d_aligned[i] and 
            close[i] <= lower_bb_1d_aligned[i] and 
            volume[i] > vol_threshold[i] and
            atr[i] > 0.01 * close[i]):  # avoid extremely low volatility
            signals[i] = 0.25
        
        # Short conditions:
        # 1. Price above daily EMA50 (bullish trend)
        # 2. Price touches or breaks above daily upper Bollinger Band (overbought)
        # 3. Volume confirmation
        # 4. ATR filter
        elif (close[i] > ema_50_1d_aligned[i] and 
              close[i] >= upper_bb_1d_aligned[i] and 
              volume[i] > vol_threshold[i] and
              atr[i] > 0.01 * close[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back to the EMA50 (mean reversion to trend)
        elif i > 0:
            if (signals[i-1] == 0.25 and close[i] >= ema_50_1d_aligned[i]):
                signals[i] = 0.0
            elif (signals[i-1] == -0.25 and close[i] <= ema_50_1d_aligned[i]):
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyEMA50_BB_MeanReversion"
timeframe = "4h"
leverage = 1.0