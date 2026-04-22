#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for ATR and volume - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(20)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR and volume average to 1d timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Calculate daily high and low for Donchian channel (20-period)
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day high with volume spike and ATR filter
            if (close[i] > high_20_aligned[i] and 
                volume[i] > 2.0 * vol_avg_aligned[i] and
                atr_aligned[i] > 0):  # Ensure volatility is present
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with volume spike and ATR filter
            elif (close[i] < low_20_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_aligned[i] and
                  atr_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: ATR-based trailing stop
            exit_signal = False
            
            if position == 1:
                # Exit long: price drops below entry price minus 2*ATR
                if close[i] <= (high_20_aligned[i] - 2.0 * atr_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises above entry price plus 2*ATR
                if close[i] >= (low_20_aligned[i] + 2.0 * atr_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_20_ATR_Volume_Breakout"
timeframe = "1d"
leverage = 1.0