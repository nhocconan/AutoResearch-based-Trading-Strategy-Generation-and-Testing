#!/usr/bin/env python3
"""
4h_RSI_Overbought_Oversold_With_Volume_Filter
Hypothesis: In ranging markets, RSI extremes often reverse. In trending markets, 
extremes can continue but volatility increases. Using 4h RSI with volume 
confirmation helps filter false signals. Works in bull markets by buying oversold 
dips in uptrends and selling overbought rallies in downtrends. Works in bear 
markets by selling bounces in downtrends and buying dips in uptrends. Volume 
ensures institutional participation, reducing whipsaws. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1-day data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1-day EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4-hour RSI(14) ===
    close = prices['close'].values
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nanmean(arr[1:period]) if period > 1 else arr[0]
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]) or np.isnan(arr[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period_rsi = 14
    avg_gain = wilder_smooth(gain, period_rsi)
    avg_loss = wilder_smooth(loss, period_rsi)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with close
    
    # === 4-hour Volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_1d_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price above daily EMA50 (uptrend) + RSI oversold + volume
            if (price_close > ema_trend and
                rsi_val < 30 and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA50 (downtrend) + RSI overbought + volume
            elif (price_close < ema_trend and
                  rsi_val > 70 and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when RSI reaches opposite extreme
            if position == 1 and rsi_val > 70:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_RSI_Overbought_Oversold_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0