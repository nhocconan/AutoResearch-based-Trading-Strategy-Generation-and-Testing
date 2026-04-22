#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily data for price action
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian channel (20-period)
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (primary)
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    highest_20_1d = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_1d = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Daily ATR(14) for volatility filter and stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20  # Strong volume surge
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d[i]) or np.isnan(highest_20_1d[i]) or np.isnan(lowest_20_1d[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume surge, above weekly EMA50
            if (close[i] > highest_20_1d[i] and vol_surge[i] and close[i] > ema_50_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume surge, below weekly EMA50
            elif (close[i] < lowest_20_1d[i] and vol_surge[i] and close[i] < ema_50_1d[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses opposite Donchian level or volatility drops significantly
            if position == 1:
                if close[i] < lowest_20_1d[i] or atr[i] < 0.5 * atr[i-1]:  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > highest_20_1d[i] or atr[i] < 0.5 * atr[i-1]:  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_WeeklyEMA50_Trend_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0