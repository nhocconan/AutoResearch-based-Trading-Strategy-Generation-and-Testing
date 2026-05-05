#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation
# Long when: RSI(14) < 30 (oversold) AND price > 4h EMA50 (uptrend) AND volume > 1.5x 20-period average AND session filter (08-20 UTC)
# Short when: RSI(14) > 70 (overbought) AND price < 4h EMA50 (downtrend) AND volume > 1.5x 20-period average AND session filter (08-20 UTC)
# Exit when: RSI crosses back to neutral (40 for long exit, 60 for short exit) OR price crosses 4h EMA50
# Uses 1h primary timeframe with 4h HTF for trend filter to capture mean reversion in trending markets
# Discrete sizing (0.20) to limit fee drag and manage drawdown
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag
# RSI mean reversion works in both bull and bear markets when combined with trend filter and volume confirmation

name = "1h_RSI_MeanReversion_4hEMA50_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h data
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Session filter: 08-20 UTC (pre-compute hours once)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI < 30 (oversold) AND price > 4h EMA50 (uptrend) 
            #              AND volume spike AND session filter
            if (rsi_values[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI > 70 (overbought) AND price < 4h EMA50 (downtrend)
            #               AND volume spike AND session filter
            elif (rsi_values[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI crosses above 40 (mean reversion complete) OR price < 4h EMA50 (trend flip)
            if rsi_values[i] > 40 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI crosses below 60 (mean reversion complete) OR price > 4h EMA50 (trend flip)
            if rsi_values[i] < 60 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals