#!/usr/bin/env python3
"""
1d_WeeklyTrend_DailyMomentum_With_Volume
Hypothesis: Use weekly trend direction from EMA34 on 1W timeframe to filter trades, 
enter on daily RSI pullbacks with volume confirmation, and exit on opposite RSI extreme. 
This captures momentum in trending markets while avoiding counter-trend trades. 
Weekly trend filter reduces whipsaws in sideways markets. Target: 10-25 trades/year.
Works in bull markets by catching pullbacks in uptrends and in bear markets by 
shorting bounces in downtrends. Volume confirmation ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # === Weekly EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily RSI(14) ===
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
    
    # === Daily Volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1w_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: weekly uptrend + RSI oversold bounce + volume
            if (price_close > ema_trend and
                rsi_val < 30 and
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + RSI overbought bounce + volume
            elif (price_close < ema_trend and
                  rsi_val > 70 and
                  vol_ratio_val > 1.3):
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

name = "1d_WeeklyTrend_DailyMomentum_With_Volume"
timeframe = "1d"
leverage = 1.0