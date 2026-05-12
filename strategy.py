#!/usr/bin/env python3
# 6h_MultiTimeframe_RSI_Momentum_Trend
# Hypothesis: Combine RSI momentum on 6h with 1d trend filter and volume confirmation.
# Long when RSI(6h) > 55 and rising, price > 1d EMA50, and volume > 1.5x average.
# Short when RSI(6h) < 45 and falling, price < 1d EMA50, and volume > 1.5x average.
# Exit when RSI crosses back to 50 level. Designed to work in both bull and bear markets
# by following the daily trend while using 6s momentum for entry timing.

name = "6h_MultiTimeframe_RSI_Momentum_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # RSI slope for momentum confirmation (3-period change)
    rsi_slope = pd.Series(rsi_values).diff(3).values
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(rsi_slope[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        rsi_val = rsi_values[i]
        rsi_slope_val = rsi_slope[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: RSI > 55 and rising, price above daily EMA50, volume > 1.5x average
            if rsi_val > 55 and rsi_slope_val > 0 and close[i] > ema_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 45 and falling, price below daily EMA50, volume > 1.5x average
            elif rsi_val < 45 and rsi_slope_val < 0 and close[i] < ema_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI falls back to 50 or below
            if rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI rises back to 50 or above
            if rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals