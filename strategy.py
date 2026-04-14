#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    open_time = pd.to_datetime(prices['open_time'])
    hours = open_time.dt.hour.values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14-period) for mean reversion
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6h ATR (14-period) for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h EMA(20) for trend filter
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(atr[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Calculate average volume for confirmation (20-period)
        vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol > (vol_avg * 1.5) if not np.isnan(vol_avg) else False
        
        # Trend filter: price > EMA20 for long, price < EMA20 for short
        trend_filter_long = price > ema_20[i]
        trend_filter_short = price < ema_20[i]
        
        # Mean reversion filter: Williams %R extremes
        # Oversold: Williams %R < -80 -> potential long
        # Overbought: Williams %R > -20 -> potential short
        wr_oversold = williams_r_aligned[i] < -80
        wr_overbought = williams_r_aligned[i] > -20
        
        if position == 0:
            # Long setup: oversold on 1d Williams %R + uptrend on 6h + volume confirmation
            if (wr_oversold and trend_filter_long and vol_confirm and vol_filter):
                position = 1
                signals[i] = position_size
            # Short setup: overbought on 1d Williams %R + downtrend on 6h + volume confirmation
            elif (wr_overbought and trend_filter_short and vol_confirm and vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns from oversold or trend breaks
            if williams_r_aligned[i] > -50 or price < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns from overbought or trend breaks
            if williams_r_aligned[i] < -50 or price > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dWilliamsMR_Trend_Filter"
timeframe = "6h"
leverage = 1.0