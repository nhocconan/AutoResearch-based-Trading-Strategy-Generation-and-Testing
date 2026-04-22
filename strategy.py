#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Weekly data for trend filter (primary trend)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA8 for trend filter
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    # Align to 6h
    ema_8_1w_6h = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    # Daily data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculation (standard)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Daily EMA21 for dynamic support/resistance
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align daily levels to 6h
    pivot_1d_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    ema_21_1d_6h = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # 6h ATR(14) for volatility filter and stop
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
    vol_surge = prices['volume'].values > 1.5 * vol_ma20  # Moderate volume surge
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_8_1w_6h[i]) or np.isnan(pivot_1d_6h[i]) or np.isnan(ema_21_1d_6h[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA8 (uptrend) and bounces from daily EMA21 support with volume
            if (close[i] > ema_8_1w_6h[i] and 
                close[i] > ema_21_1d_6h[i] and 
                low[i] <= ema_21_1d_6h[i] * 1.005 and  # Touched or slightly below support
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA8 (downtrend) and rejects from daily EMA21 resistance with volume
            elif (close[i] < ema_8_1w_6h[i] and 
                  close[i] < ema_21_1d_6h[i] and 
                  high[i] >= ema_21_1d_6h[i] * 0.995 and  # Touched or slightly above resistance
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses weekly EMA8 (trend change) or volatility drops significantly
            if position == 1:
                if close[i] < ema_8_1w_6h[i] or atr[i] < 0.4 * atr[i-1]:  # Trend change or vol drop
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema_8_1w_6h[i] or atr[i] < 0.4 * atr[i-1]:  # Trend change or vol drop
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyEMA8_Trend_DailyEMA21_Bounce_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0