#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d Trend-Following with Volume Confirmation
# Uses 4h EMA crossover as primary trend signal, confirmed by 1d RSI extreme rejection
# Volume spike filter ensures momentum validity
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag
# Works in bull/bear by following trend with momentum confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA (21) for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop for RSI filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI (14) for overbought/oversold conditions
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, np.nan, avg_loss)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike filter (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma == 0, np.nan, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Start after enough data for calculations
    start = 50  # for EMA and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above 4h EMA (uptrend) + RSI not overbought + volume spike
            if (price > ema_4h_aligned[i] and 
                rsi_1d_aligned[i] < 70 and 
                vol_ratio[i] > 1.5):
                position = 1
                signals[i] = position_size
            # Short: price below 4h EMA (downtrend) + RSI not oversold + volume spike
            elif (price < ema_4h_aligned[i] and 
                  rsi_1d_aligned[i] > 30 and 
                  vol_ratio[i] > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 4h EMA or RSI overbought
            if price < ema_4h_aligned[i] or rsi_1d_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 4h EMA or RSI oversold
            if price > ema_4h_aligned[i] or rsi_1d_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h1d_Trend_Volume_Filter"
timeframe = "1h"
leverage = 1.0