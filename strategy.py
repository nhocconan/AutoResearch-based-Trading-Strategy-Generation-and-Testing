#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3S3_Breakout_1dTrend_1hVWAP_Reversion"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute hour filter
    hours = prices.index.hour
    
    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily Camarilla levels (S3/R3)
    high_low_range = high_1d - low_1d
    camarilla_high = high_1d + 1.1 * high_low_range
    camarilla_low = low_1d - 1.1 * high_low_range
    camarilla_range = camarilla_high - camarilla_low
    R3 = camarilla_low + camarilla_range * 1.1000
    S3 = camarilla_high - camarilla_range * 1.1000
    
    # Align Camarilla levels to 1h timeframe (wait for daily close)
    R3_1h = align_htf_to_ltf(prices, df_1d, R3)
    S3_1h = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1h VWAP for mean reversion entry timing
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(typical_price, np.nan), where=vwap_den!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_1h[i]) or np.isnan(S3_1h[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if position == 0 and in_session:
            # Long: Price below S3 + below daily EMA34 + below VWAP (oversold in downtrend)
            if (close[i] < S3_1h[i] and
                close[i] < ema_34_1d_aligned[i] and
                close[i] < vwap[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price above R3 + above daily EMA34 + above VWAP (overbought in uptrend)
            elif (close[i] > R3_1h[i] and
                  close[i] > ema_34_1d_aligned[i] and
                  close[i] > vwap[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Price crosses above VWAP (mean reversion complete) or breaks R3
            if (close[i] > vwap[i] or
                close[i] > R3_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price crosses below VWAP (mean reversion complete) or breaks S3
            if (close[i] < vwap[i] or
                close[i] < S3_1h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals