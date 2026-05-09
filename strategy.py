#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Keltner_Breakout_Trend_Volume"
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
    
    # Get 1d data for Keltner calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Keltner Channels (20-period)
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    close_series = pd.Series(df_1d['close'])
    
    # EMA20 of typical price for middle line
    typical_price = (high_series + low_series + close_series) / 3
    ema20_tp = typical_price.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(20) for channel width
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr20 = tr.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and lower bands
    keltner_upper = ema20_tp + (2.0 * atr20)
    keltner_lower = ema20_tp - (2.0 * atr20)
    
    # Align to 1d
    keltner_upper_1d = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_1d = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema20_tp_1d = align_htf_to_ltf(prices, df_1d, ema20_tp)
    
    # Trend filter: 20-week EMA on close
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_1d = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: current volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'])
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.5)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need enough data for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_upper_1d[i]) or np.isnan(keltner_lower_1d[i]) or
            np.isnan(ema20_tp_1d[i]) or np.isnan(ema20_1w_1d[i]) or
            np.isnan(volume_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = keltner_upper_1d[i]
        lower = keltner_lower_1d[i]
        middle = ema20_tp_1d[i]
        trend = ema20_1w_1d[i]
        vol_filter = volume_filter_1d_aligned[i]
        
        if position == 0:
            # Enter long: break above upper band with volume and above weekly trend
            if close[i] > upper and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower band with volume and below weekly trend
            elif close[i] < lower and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle line (mean reversion)
            if close[i] < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle line (mean reversion)
            if close[i] > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals