#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-calculate hour filter
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data once
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA21 and EMA50 for trend direction
    close_4h_series = pd.Series(close_4h)
    ema21_4h = close_4h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 1h
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for long-term trend
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h RSI(14) for entry timing
    close_series = pd.Series(prices['close'])
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 1h volume filter
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    start = max(50, 200)  # Need enough data for EMA200
    
    for i in range(start, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data is NaN
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close_series.iloc[i]
        
        # Trend alignment: 4h EMA21 > EMA50 AND price > 1d EMA200 for long
        # 4h EMA21 < EMA50 AND price < 1d EMA200 for short
        bullish = ema21_4h_aligned[i] > ema50_4h_aligned[i] and price > ema200_1d_aligned[i]
        bearish = ema21_4h_aligned[i] < ema50_4h_aligned[i] and price < ema200_1d_aligned[i]
        
        if position == 0:
            if bullish and rsi[i] < 40 and volume[i] > 1.5 * vol_ma[i]:
                position = 1
                signals[i] = position_size
            elif bearish and rsi[i] > 60 and volume[i] > 1.5 * vol_ma[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: 4h EMA21 < EMA50 OR RSI > 70
            if ema21_4h_aligned[i] < ema50_4h_aligned[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: 4h EMA21 > EMA50 OR RSI < 30
            if ema21_4h_aligned[i] > ema50_4h_aligned[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_EMA_RSI_Volume_Filter"
timeframe = "1h"
leverage = 1.0