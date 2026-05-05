#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R4 AND close > EMA50(12h) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S4 AND close < EMA50(12h) AND volume > 2.0x 20-period average
# Exit when price retracement to Camarilla pivot point (PP) OR EMA50(12h) trend flip
# Uses 4h primary timeframe with 12h HTF for trend filter to reduce whipsaw and avoid overtrading
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag
# Camarilla levels from daily OHLC provide intraday structure; breakouts with volume and trend filter capture strong moves

name = "4h_Camarilla_R4_S4_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get daily data for Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily OHLC (using previous day's values to avoid look-ahead)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla: PP = (H+L+C)/3, R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r4_1d = close_1d + (high_1d - low_1d) * 1.1
    s4_1d = close_1d - (high_1d - low_1d) * 1.1
    
    # Align to 4h timeframe (using previous day's levels to avoid look-ahead)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 AND close > EMA50(12h) AND volume spike
            if (high[i] > r4_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S4 AND close < EMA50(12h) AND volume spike
            elif (low[i] < s4_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Camarilla pivot point OR close < EMA50(12h) (trend flip)
            if close[i] <= pp_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Camarilla pivot point OR close > EMA50(12h) (trend flip)
            if close[i] >= pp_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals