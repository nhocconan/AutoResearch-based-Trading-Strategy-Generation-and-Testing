#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R3S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily ATR(14) for volume filter
    tr1 = high[1:] - low[:-1]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[:-1] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr14).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr14 > 1.5 * atr_ma20
    
    # Weekly trend filter: EMA34 on 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from previous day's data
    # Using previous day's OHLC to avoid look-ahead
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    # Camarilla R3, R4, S3, S4 levels
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.0 * camarilla_range
    r4 = prev_close + 1.5 * camarilla_range
    s3 = prev_close - 1.0 * camarilla_range
    s4 = prev_close - 1.5 * camarilla_range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(atr14[i]) or np.isnan(atr_ma20[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume and weekly uptrend
            if (price > r3[i] and 
                vol_filter[i] and 
                price > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below S3 with volume and weekly downtrend
            elif (price < s3[i] and 
                  vol_filter[i] and 
                  price < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price retreats below R3 or weekly trend fails
            if (price < r3[i] or 
                price < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above S3 or weekly trend fails
            if (price > s3[i] or 
                price > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals