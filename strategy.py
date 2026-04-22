#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Band Squeeze + RSI Mean Reversion with Volume Spike and 1-Day Trend Filter.
Long when Bollinger Bands are squeezed (low volatility), RSI < 30, price touches lower band, volume spike, and 1d EMA34 is rising.
Short when Bollinger Bands are squeezed, RSI > 70, price touches upper band, volume spike, and 1d EMA34 is falling.
Exit when RSI crosses 50 or Bollinger Band width expands beyond 20-day average.
Designed for low trade frequency by requiring volatility contraction, extreme RSI, volume confirmation, and trend alignment.
Works in ranging markets (mean reversion) and avoids trending markets via Bollinger Band width filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bandwidth = (upper - lower) / sma  # Normalized bandwidth
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Bollinger Band squeeze: bandwidth < 20-period average bandwidth
        bw_ma_20 = pd.Series(bandwidth).rolling(window=20, min_periods=20).mean().values
        bw_squeeze = bandwidth[i] < bw_ma_20[i] if not np.isnan(bw_ma_20[i]) else False
        
        # Volume spike
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bollinger squeeze, RSI < 30, price <= lower band, volume spike, 1d EMA34 rising
            if (bw_squeeze and rsi[i] < 30 and close[i] <= lower[i] and vol_spike and 
                i > 0 and not np.isnan(ema34_1d_aligned[i-1]) and ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger squeeze, RSI > 70, price >= upper band, volume spike, 1d EMA34 falling
            elif (bw_squeeze and rsi[i] > 70 and close[i] >= upper[i] and vol_spike and 
                  i > 0 and not np.isnan(ema34_1d_aligned[i-1]) and ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI crosses 50 or Bollinger Band width expands beyond 20-day average
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI >= 50 or bandwidth expands
                if rsi[i] >= 50 or (not np.isnan(bw_ma_20[i]) and bandwidth[i] >= bw_ma_20[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI <= 50 or bandwidth expands
                if rsi[i] <= 50 or (not np.isnan(bw_ma_20[i]) and bandwidth[i] >= bw_ma_20[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_BollingerSqueeze_RSI_MeanReversion_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0