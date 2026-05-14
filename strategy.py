#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above R1 with 12h EMA50 uptrend and 4h volume > 2.0x 20-period average.
# Short when price breaks below S1 with 12h EMA50 downtrend and 4h volume > 2.0x 20-period average.
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn and strict volume confirmation to reduce false breakouts.
# Target: 100-200 trades over 4 years (25-50/year) for 4h timeframe.
# Works in bull/bear: 12h EMA50 ensures trend alignment, Camarilla provides structure within trend.

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # 4h Volume confirmation: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_bullish = close_12h > ema_50  # Bullish if price above EMA50
    ema_50_bearish = close_12h < ema_50  # Bearish if price below EMA50
    
    # Align 12h indicators to 4h
    ema_50_bullish_aligned = align_htf_to_ltf(prices, df_12h, ema_50_bullish.astype(float))
    ema_50_bearish_aligned = align_htf_to_ltf(prices, df_12h, ema_50_bearish.astype(float))
    
    # --- 4h Camarilla Pivot Points (Prior Day OHLC) ---
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    df_1d_pivot = get_htf_data(prices, '1d')
    if len(df_1d_pivot) > 0:
        # Map each 4h bar to prior day's OHLC
        open_time = prices['open_time']
        prior_day_start = open_time - pd.Timedelta(days=1)
        prior_day_start = prior_day_start.dt.normalize()  # Start of prior day
        
        # Create series of prior day data aligned to 4h bars
        for i in range(n):
            pd_ts = prior_day_start.iloc[i]
            day_mask = (df_1d_pivot['open_time'] >= pd_ts) & (df_1d_pivot['open_time'] < pd_ts + pd.Timedelta(days=1))
            if day_mask.any():
                day_data = df_1d_pivot.loc[day_mask]
                high_val = day_data['high'].iloc[0]
                low_val = day_data['low'].iloc[0]
                close_val = day_data['close'].iloc[0]
                range_val = high_val - low_val
                camarilla_r1[i] = close_val + (range_val * 1.1 / 12)
                camarilla_s1[i] = close_val - (range_val * 1.1 / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_bullish_aligned[i]) or 
            np.isnan(ema_50_bearish_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + 12h uptrend + volume confirmation
            if (close[i] > camarilla_r1[i] and 
                ema_50_bullish_aligned[i] > 0.5 and 
                volume_confirm[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 12h downtrend + volume confirmation
            elif (close[i] < camarilla_s1[i] and 
                  ema_50_bearish_aligned[i] > 0.5 and 
                  volume_confirm[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1
            if close[i] < camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1
            if close[i] > camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals