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
    
    # Get 4h data for trend and volatility filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 4h ATR14 for volatility filter
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr14_4h)
    
    # Volume filter: require volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for EMA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(atr14_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > EMA20 + volume + moderate volatility
            if (close[i] > ema20_4h_aligned[i] and 
                volume_filter[i] and 
                atr14_4h_aligned[i] > 0.5 * np.nanmedian(atr14_4h_aligned[:i+1]) and
                atr14_4h_aligned[i] < 2.0 * np.nanmedian(atr14_4h_aligned[:i+1]) and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price < EMA20 + volume + moderate volatility
            elif (close[i] < ema20_4h_aligned[i] and 
                  volume_filter[i] and 
                  atr14_4h_aligned[i] > 0.5 * np.nanmedian(atr14_4h_aligned[:i+1]) and
                  atr14_4h_aligned[i] < 2.0 * np.nanmedian(atr14_4h_aligned[:i+1]) and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < EMA20 (trend change) or volatility too high
            if (close[i] < ema20_4h_aligned[i] or 
                atr14_4h_aligned[i] > 2.5 * np.nanmedian(atr14_4h_aligned[:i+1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price > EMA20 (trend change) or volatility too high
            if (close[i] > ema20_4h_aligned[i] or 
                atr14_4h_aligned[i] > 2.5 * np.nanmedian(atr14_4h_aligned[:i+1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA20_Vol_ModerateVol_Filter"
timeframe = "1h"
leverage = 1.0