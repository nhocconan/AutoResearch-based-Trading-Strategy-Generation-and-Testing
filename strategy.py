#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Keltner_Channel_Breakout_1wTrend"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Keltner Channel on 1d
    ema_20_1d = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10 = pd.Series(np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])).ewm(span=10, adjust=False, min_periods=10).mean()
    atr_10 = np.concatenate([np.full(10, np.nan), atr_10.values[:-1]])
    
    upper_keltner = ema_20_1d + (2.0 * atr_10)
    lower_keltner = ema_20_1d - (2.0 * atr_10)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or
            np.isnan(ema_20_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_20_1w_aligned[i]
        upper = upper_keltner[i]
        lower = lower_keltner[i]
        vol_spike = volume_spike[i]
        ema_1d = ema_20_1d[i]
        
        if position == 0:
            # Enter long: Close breaks above upper Keltner with 1w uptrend and volume spike
            if close[i] > upper and close[i] > ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below lower Keltner with 1w downtrend and volume spike
            elif close[i] < lower and close[i] < ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close below EMA(20) or trend breaks (price < 1w EMA)
            if close[i] < ema_1d or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close above EMA(20) or trend breaks (price > 1w EMA)
            if close[i] > ema_1d or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals