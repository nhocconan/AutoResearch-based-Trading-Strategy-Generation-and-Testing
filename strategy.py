#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h ADX trend filter and 1d Williams %R mean reversion.
# Long: Price crosses above 4h SMA50 + 12h ADX > 25 + 1d Williams %R < -80 (oversold).
# Short: Price crosses below 4h SMA50 + 12h ADX > 25 + 1d Williams %R > -20 (overbought).
# Exit: Price crosses back through SMA50 or ADX drops below 20.
# Uses 4h for primary signal, 12h for trend strength filter, 1d for mean reversion timing.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Target: 50-150 total trades over 4 years (12-38/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 4h data for SMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    sma_50_4h = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    
    # 12h data for ADX trend strength
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros(len(high))
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr
        
        dx = np.zeros(len(high))
        for i in range(len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_14_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # 1d data for Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Align indicators to 4h timeframe
    sma_50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_50_4h)
    adx_14_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_14_12h)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(sma_50_4h_aligned[i]) or np.isnan(adx_14_12h_aligned[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        sma = sma_50_4h_aligned[i]
        adx = adx_14_12h_aligned[i]
        williams = williams_r_aligned[i]
        
        if position == 0:
            # Long: price crosses above SMA50 + strong trend (ADX>25) + oversold (Williams%R<-80)
            if (price > sma and 
                adx > 25 and
                williams < -80):
                position = 1
                signals[i] = position_size
            # Short: price crosses below SMA50 + strong trend (ADX>25) + overbought (Williams%R>-20)
            elif (price < sma and 
                  adx > 25 and
                  williams > -20):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below SMA50 or trend weakens (ADX<20)
            if (price < sma or
                adx < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above SMA50 or trend weakens (ADX<20)
            if (price > sma or
                adx < 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_1d_SMA50_ADX_WilliamsR"
timeframe = "4h"
leverage = 1.0