#!/usr/bin/env python3
name = "1h_RSI_Stoch_Bollinger_Confluence_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend filter
    ema_21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Get 1d data for regime filter (Bollinger Band width)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Band width
    ma_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    bb_width_1d = (std_20_1d * 2) / ma_20_1d
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Calculate 1h RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1h Stochastic (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Calculate 1h Bollinger Bands (20,2)
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00 - 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure BB, Stoch, and volume data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(bb_width_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(d_percent[i]) or
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 1.5 x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long conditions: Oversold + BB bounce + 4h uptrend + low volatility regime
            if (rsi[i] < 30 and 
                d_percent[i] < 20 and 
                close[i] < lower_bb[i] and
                close[i] > ema_21_4h_aligned[i] and  # 4h uptrend filter
                bb_width_1d_aligned[i] < 0.05 and   # Low volatility regime (BB width < 5%)
                volume_filter):
                signals[i] = 0.20
                position = 1
            # Short conditions: Overbought + BB rejection + 4h downtrend + low volatility regime
            elif (rsi[i] > 70 and 
                  d_percent[i] > 80 and 
                  close[i] > upper_bb[i] and
                  close[i] < ema_21_4h_aligned[i] and  # 4h downtrend filter
                  bb_width_1d_aligned[i] < 0.05 and   # Low volatility regime
                  volume_filter):
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit conditions: Mean reversion or trend exhaustion
            if position == 1:  # Long position
                # Exit when RSI reaches overbought or price touches upper BB
                if rsi[i] > 70 or close[i] > upper_bb[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # Short position
                # Exit when RSI reaches oversold or price touches lower BB
                if rsi[i] < 30 or close[i] < lower_bb[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals