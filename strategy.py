#!/usr/bin/env python3
name = "6h_RSI_Divergence_1wTrend_1dVol"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== 1w Trend Filter (HTF) =====
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ===== 1d Volume Spike Filter =====
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # ===== 6h RSI (14) =====
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ===== 6h RSI Swing Detection (for divergence) =====
    lookback = 10
    rsi_high = np.full(n, np.nan)
    rsi_low = np.full(n, np.nan)
    price_high = np.full(n, np.nan)
    price_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window_rsi = rsi[i-lookback:i+1]
        window_price = close[i-lookback:i+1]
        if np.nanmax(window_rsi) == rsi[i]:
            rsi_high[i] = rsi[i]
            price_high[i] = close[i]
        if np.nanmin(window_rsi) == rsi[i]:
            rsi_low[i] = rsi[i]
            price_low[i] = close[i]
    
    # Align HTF indicators
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(price_high[i]) or np.isnan(price_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: RSI makes higher low, price makes lower low
            if (i >= 20 and 
                not np.isnan(rsi_low[i-10]) and not np.isnan(rsi_low[i]) and
                rsi_low[i] > rsi_low[i-10] and  # Higher low in RSI
                not np.isnan(price_low[i-10]) and not np.isnan(price_low[i]) and
                price_low[i] < price_low[i-10] and  # Lower low in price
                close[i] > ema50_1w_aligned[i] and  # Above weekly trend
                vol_spike_1d_aligned[i] > 0.5):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Bearish divergence: RSI makes lower high, price makes higher high
            elif (i >= 20 and 
                  not np.isnan(rsi_high[i-10]) and not np.isnan(rsi_high[i]) and
                  rsi_high[i] < rsi_high[i-10] and  # Lower high in RSI
                  not np.isnan(price_high[i-10]) and not np.isnan(price_high[i]) and
                  price_high[i] > price_high[i-10] and  # Higher high in price
                  close[i] < ema50_1w_aligned[i] and  # Below weekly trend
                  vol_spike_1d_aligned[i] > 0.5):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or price below weekly EMA
            if rsi[i] > 70 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold or price above weekly EMA
            if rsi[i] < 30 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals