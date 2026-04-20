#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R reversal with 4h EMA trend filter and volume confirmation
# - Williams %R(14) on 1h: long when < -80 (oversold), short when > -20 (overbought)
# - Filter: only trade in direction of 4h EMA(34) trend
# - Volume filter: require volume > 1.3x 20-period average
# - Exit: opposite Williams %R signal or trailing stop via signal reduction
# - Session filter: 08-20 UTC to avoid low-volume periods
# - Target: 20-40 trades per year per symbol (80-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(34)
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R(14) on 1h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(williams_r[i]) or np.isnan(ema_34_4h_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        vol = volume[i]
        ema_trend = ema_34_4h_aligned[i]
        
        if position == 0 and in_session:
            # Long entry: Williams %R oversold (< -80) + above 4h EMA + volume surge
            if williams_r[i] < -80 and price > ema_trend and vol > 1.3 * vol_ma[i]:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short entry: Williams %R overbought (> -20) + below 4h EMA + volume surge
            elif williams_r[i] > -20 and price < ema_trend and vol > 1.3 * vol_ma[i]:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: Williams %R overbought OR price drops below 4h EMA
            if williams_r[i] > -20 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Williams %R oversold OR price rises above 4h EMA
            if williams_r[i] < -80 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_WilliamsR_4hEMAFilter_Volume_Session"
timeframe = "1h"
leverage = 1.0