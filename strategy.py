#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WickReversal_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend: EMA50 on 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.5x 10-period SMA
    vol_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_filter = volume > 1.5 * vol_ma10
    
    # Wick rejection: upper/lower wick > 60% of body
    body = np.abs(close - prices['open'].values)
    upper_wick = high - np.maximum(close, prices['open'].values)
    lower_wick = np.minimum(close, prices['open'].values) - low
    upper_wick_ratio = np.where(body > 0, upper_wick / body, 0)
    lower_wick_ratio = np.where(body > 0, lower_wick / body, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma10[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        open_price = prices['open'].values[i]
        
        if position == 0:
            # Long: bullish engulfing or hammer with weekly uptrend and volume
            bullish_engulfing = (close > open_price and 
                                 close > prices['open'].values[i-1] and 
                                 open_price < prices['close'].values[i-1])
            hammer = (lower_wick_ratio[i] > 0.6 and 
                      body[i] > 0 and 
                      upper_wick_ratio[i] < 0.3)
            
            if ((bullish_engulfing or hammer) and 
                price > ema50_1w_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: bearish engulfing or shooting star with weekly downtrend and volume
            bearish_engulfing = (close < open_price and 
                                 close < prices['open'].values[i-1] and 
                                 open_price > prices['close'].values[i-1])
            shooting_star = (upper_wick_ratio[i] > 0.6 and 
                             body[i] > 0 and 
                             lower_wick_ratio[i] < 0.3)
            
            if ((bearish_engulfing or shooting_star) and 
                price < ema50_1w_aligned[i] and 
                vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price crosses below weekly EMA or loses volume
            if (price < ema50_1w_aligned[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly EMA or loses volume
            if (price > ema50_1w_aligned[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals