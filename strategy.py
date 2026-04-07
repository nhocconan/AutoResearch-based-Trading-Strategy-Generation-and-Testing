#!/usr/bin/env python3
"""
1h_bollinger_squeeze_4h1d_trend_volume_v1
Hypothesis: On 1h timeframe, use Bollinger Band squeeze (low volatility breakout) with 4h trend filter (EMA50/EMA200) and 1d volume confirmation. Enter long when price breaks above upper Bollinger Band with 4h EMA50 > EMA200 and 1d volume > 1.5x average; enter short when price breaks below lower Bollinger Band with 4h EMA50 < EMA200 and 1d volume > 1.5x average. Exit when price returns to middle Bollinger Band or trend reverses. Bollinger squeeze captures low volatility breakouts that work in both bull and bear markets, while 4h trend filter ensures we trade with the higher timeframe momentum. Volume confirmation avoids false breakouts. Targets 15-30 trades/year to minimize fee drag on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_bollinger_squeeze_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 1h
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # 4h EMA50 and EMA200 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_4h = pd.Series(df_4h['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(basis[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or vol_ma_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-day average 1d volume
        vol_confirm = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price returns to middle Bollinger Band (mean reversion)
            if close[i] <= basis[i]:
                exit_long = True
            # Exit if 4h trend reverses (EMA50 < EMA200)
            elif ema50_4h_aligned[i] < ema200_4h_aligned[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price returns to middle Bollinger Band (mean reversion)
            if close[i] >= basis[i]:
                exit_short = True
            # Exit if 4h trend reverses (EMA50 > EMA200)
            elif ema50_4h_aligned[i] > ema200_4h_aligned[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Bollinger Band with 4h uptrend and volume confirmation
            long_entry = False
            if (close[i] > upper[i] and close[i-1] <= upper[i-1] and
                ema50_4h_aligned[i] > ema200_4h_aligned[i] and vol_confirm):
                long_entry = True
            
            # Short entry: price breaks below lower Bollinger Band with 4h downtrend and volume confirmation
            short_entry = False
            if (close[i] < lower[i] and close[i-1] >= lower[i-1] and
                ema50_4h_aligned[i] < ema200_4h_aligned[i] and vol_confirm):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.20
            elif short_entry:
                position = -1
                signals[i] = -0.20
    
    return signals