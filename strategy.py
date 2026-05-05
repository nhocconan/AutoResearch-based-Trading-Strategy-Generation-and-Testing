#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w Trend Filter and Volume Confirmation
# Long when: Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND 1w close > 1w EMA34 AND 1d volume > 1.5x 20-period average
# Short when: Alligator jaws < teeth < lips AND 1w close < 1w EMA34 AND 1d volume > 1.5x 20-period average
# Exit when Alligator lines re-cross (jaws crosses teeth) indicating trend exhaustion
# Williams Alligator identifies trending vs ranging markets via smoothed moving averages
# 1w EMA34 filter ensures alignment with weekly trend to avoid counter-trend trades
# Volume spike confirms institutional participation in breakout
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25 to minimize fee churn

name = "1d_WilliamsAlligator_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d average volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator: Smoothed Moving Average (SMMA) with specific periods
    # SMMA calculation: today's SMMA = (yesterday's SMMA * (period-1) + today's price) / period
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Lips (5), Teeth (8), Jaws (13) - all SMMA of median price
    median_price = (high + low) / 2
    lips = smma(median_price, 5)   # Green line
    teeth = smma(median_price, 8)  # Red line
    jaws = smma(median_price, 13)  # Blue line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after Alligator warmup
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Jaws > Teeth > Lips (bullish alignment) + 1w uptrend + volume spike
            if (jaws[i] > teeth[i] and teeth[i] > lips[i] and 
                close[i] > ema_34_aligned[i] and 
                volume[i] > 1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaws < Teeth < Lips (bearish alignment) + 1w downtrend + volume spike
            elif (jaws[i] < teeth[i] and teeth[i] < lips[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Jaw crosses below Teeth (trend weakening) or stoploss
            if jaws[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Jaw crosses above Teeth (trend weakening) or stoploss
            if jaws[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals