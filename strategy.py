#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND close > EMA50(12h) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S3 AND close < EMA50(12h) AND volume > 2.0x 20-period average
# Exit when price retraces to Camarilla pivot point (PP) OR close crosses EMA50(12h) (trend flip)
# Uses 4h primary timeframe with 12h HTF for trend filter to capture multi-day moves with controlled frequency
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Camarilla R3/S3 levels provide strong intraday resistance/support; breakouts with volume and trend filter
# capture institutional participation while avoiding false breakouts in choppy markets

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_Volume"
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
    
    # Get 12h data ONCE before loop for EMA50 trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from 12h OHLC (using previous 12h bar's values to avoid look-ahead)
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    r3_12h = close_12h + (high_12h - low_12h) * 1.1 / 4.0
    s3_12h = close_12h - (high_12h - low_12h) * 1.1 / 4.0
    
    # Align to 4h timeframe (using previous 12h bar's levels to avoid look-ahead)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
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
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND close > EMA50(12h) AND volume spike
            if (high[i] > r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND close < EMA50(12h) AND volume spike
            elif (low[i] < s3_aligned[i] and 
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