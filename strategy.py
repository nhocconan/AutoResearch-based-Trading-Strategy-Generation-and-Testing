#!/usr/bin/env python3
# 4h_1d_cci_rsi_volume_v1
# Strategy: 4h CCI + RSI momentum with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions while RSI confirms momentum strength.
# Volume filters ensure institutional participation. 1d EMA50 trend filter prevents counter-trend trades.
# Designed for low frequency (20-40 trades/year) to minimize fee drag in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # CCI calculation: (Typical Price - SMA) / (0.015 * Mean Deviation)
    typical_price = (high + low + close) / 3
    tp_series = pd.Series(typical_price)
    sma_20 = tp_series.rolling(window=20, min_periods=20).mean()
    mean_dev = tp_series.rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
    )
    cci = (tp_series - sma_20) / (0.015 * mean_dev)
    cci = cci.values
    
    # RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(cci[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: CCI momentum + RSI confirmation + volume + trend alignment
        if (cci[i] > 100 and rsi[i] > 50 and  # CCI overbought + RSI bullish
            vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (cci[i] < -100 and rsi[i] < 50 and  # CCI oversold + RSI bearish
              vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: CCI mean reversion or trend change
        elif position == 1 and (cci[i] < 0 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci[i] > 0 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals