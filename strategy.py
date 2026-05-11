#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
# Hypothesis: Use Camarilla R1/S1 breakout on 1h with 4h trend and 1d volume confirmation.
# In bull markets: long when 4h uptrend + price breaks above R1 + 1d volume above average.
# In bear markets: short when 4h downtrend + price breaks below S1 + 1d volume above average.
# Volume filter reduces false breakouts, trend filter ensures directional bias.
# Camarilla levels provide intraday support/resistance; breakouts often lead to sustained moves.
# Target: 15-37 trades/year on 1h timeframe with strict entry conditions.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h EMA50 for trend direction ---
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_slope = ema_50_4h - np.roll(ema_50_4h, 1)
    ema_50_4h_slope[0] = 0
    ema_50_4h_slope = pd.Series(ema_50_4h_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h_slope)
    
    # --- 1d volume average for confirmation ---
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # --- Calculate Camarilla levels for 1h (using previous bar's OHLC) ---
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We use previous bar's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    rang = prev_high - prev_low
    R1 = prev_close + 1.1 * rang / 12
    S1 = prev_close - 1.1 * rang / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Session filter: 8-20 UTC (inclusive)
    hours = prices.index.hour
    
    # Warmup: enough for EMA50 (50) and Camarilla calculation
    start_idx = 50
    
    for i in range(start_idx, n):
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_slope_aligned[i]) or
            np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(R1[i]) or
            np.isnan(S1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 4h EMA50 slope
        uptrend = ema_50_4h_slope_aligned[i] > 0
        downtrend = ema_50_4h_slope_aligned[i] < 0
        
        # Volume confirmation: current 1h volume > 1d average volume
        vol_surge = volume[i] > vol_avg_1d_aligned[i]
        
        if position == 0 and in_session:
            if uptrend and vol_surge:
                # Long: 4h uptrend + volume surge + price breaks above R1
                if close[i] > R1[i]:
                    signals[i] = 0.20
                    position = 1
            elif downtrend and vol_surge:
                # Short: 4h downtrend + volume surge + price breaks below S1
                if close[i] < S1[i]:
                    signals[i] = -0.20
                    position = -1
        else:
            if position == 1:
                # Exit long: 4h trend turns down OR price closes below S1 (reversion)
                if downtrend or close[i] < S1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: 4h trend turns up OR price closes above R1 (reversion)
                if uptrend or close[i] > R1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals