#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 60:
        return np.zeros(n)
    
    # 1d RSI(14) for momentum
    close_daily = pd.Series(df_daily['close'].values)
    delta = close_daily.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi14_daily = (100 - (100 / (1 + rs))).values
    rsi14_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi14_daily)
    
    # 1d ADX(14) for trend strength
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily_arr = df_daily['close'].values
    
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily_arr, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.where((high_daily - np.roll(high_daily, 1)) > (np.roll(low_daily, 1) - low_daily), 
                       np.maximum(high_daily - np.roll(high_daily, 1), 0), 0)
    minus_dm = np.where((np.roll(low_daily, 1) - low_daily) > (high_daily - np.roll(high_daily, 1)), 
                        np.maximum(np.roll(low_daily, 1) - low_daily, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx14_daily = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx14_daily_aligned = align_htf_to_ltf(prices, df_daily, adx14_daily)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi14_daily_aligned[i]) or np.isnan(adx14_daily_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI > 50 (bullish momentum) + ADX > 25 (strong trend) + volume filter
            if (rsi14_daily_aligned[i] > 50 and 
                adx14_daily_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 50 (bearish momentum) + ADX > 25 (strong trend) + volume filter
            elif (rsi14_daily_aligned[i] < 50 and 
                  adx14_daily_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI < 50 (momentum shift) OR ADX < 20 (weakening trend)
            if (rsi14_daily_aligned[i] < 50 or 
                adx14_daily_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI > 50 (momentum shift) OR ADX < 20 (weakening trend)
            if (rsi14_daily_aligned[i] > 50 or 
                adx14_daily_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_ADX_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0