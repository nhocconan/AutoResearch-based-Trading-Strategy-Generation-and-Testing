#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and reference
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA20 for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily ATR(14) for volatility filter
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day's range
    # Using previous day's close, high, low
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First value will be invalid (rolled from last), but min_periods handles this
    prev_range_1d = prev_high_1d - prev_low_1d
    
    camarilla_r4 = prev_close_1d + prev_range_1d * 1.1 / 2
    camarilla_r3 = prev_close_1d + prev_range_1d * 1.1 / 4
    camarilla_s3 = prev_close_1d - prev_range_1d * 1.1 / 4
    camarilla_s4 = prev_close_1d - prev_range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r4_12h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_12h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_20_12h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma_12h * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_12h[i]) or np.isnan(atr_14_12h[i]) or
            np.isnan(camarilla_r4_12h[i]) or np.isnan(camarilla_r3_12h[i]) or
            np.isnan(camarilla_s3_12h[i]) or np.isnan(camarilla_s4_12h[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < S3 or trend fails
            if close[i] < camarilla_s3_12h[i] or close[i] < ema_20_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R3 or trend fails
            if close[i] > camarilla_r3_12h[i] or close[i] > ema_20_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_20_12h[i]
            bearish = close[i] < ema_20_12h[i]
            
            # Volatility filter: avoid extremely low volatility
            vol_ok = atr_14_12h[i] > 0
            
            # Long: price > R4 + bullish trend + volume
            if (close[i] > camarilla_r4_12h[i] and 
                bullish and 
                vol_filter[i] and
                vol_ok):
                position = 1
                signals[i] = 0.25
            # Short: price < S4 + bearish trend + volume
            elif (close[i] < camarilla_s4_12h[i] and 
                  bearish and 
                  vol_filter[i] and
                  vol_ok):
                position = -1
                signals[i] = -0.25
    
    return signals