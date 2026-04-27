#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R reversal with 4h EMA20 trend filter and volume confirmation.
# Long when Williams %R crosses above -20 from below (oversold reversal) with 4h EMA20 uptrend and volume > 1.5x average.
# Short when Williams %R crosses below -80 from above (overbought reversal) with 4h EMA20 downtrend and volume > 1.5x average.
# Exit when Williams %R crosses back through -50 (mean reversion).
# Uses Williams %R for precise reversal timing on 1h timeframe, targeting 15-37 trades per year (60-150 over 4 years).
# Session filter: 08-20 UTC to reduce noise trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema_period = 20
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * (2 / (ema_period + 1)) + 
                         ema_4h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Williams %R (14-period)
    willr_period = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    willr = np.full(n, np.nan)
    
    for i in range(willr_period - 1, n):
        highest_high[i] = np.max(high[i - willr_period + 1:i + 1])
        lowest_low[i] = np.min(low[i - willr_period + 1:i + 1])
        if highest_high[i] != lowest_low[i]:
            willr[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
    
    # Williams %R previous value for crossover detection
    willr_prev = np.full(n, np.nan)
    willr_prev[1:] = willr[:-1]
    
    # Align 4h EMA to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need Williams %R, EMA20, and volume MA20
    start_idx = max(willr_period, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(willr[i]) or np.isnan(willr_prev[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below with 4h EMA20 uptrend and volume filter
            if (willr_prev[i] <= -20 and willr[i] > -20 and 
                price > ema_4h_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: Williams %R crosses below -80 from above with 4h EMA20 downtrend and volume filter
            elif (willr_prev[i] >= -80 and willr[i] < -80 and 
                  price < ema_4h_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses below -50 from above
            if willr_prev[i] >= -50 and willr[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R crosses above -50 from below
            if willr_prev[i] <= -50 and willr[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_WilliamsR14_Reversal_4hEMA20_Volume"
timeframe = "1h"
leverage = 1.0