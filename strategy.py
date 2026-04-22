#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Weekly data for trend filter and volatility
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly ATR(14) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]  # First value
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian(20) breakout levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian upper/lower bands (20-period)
    donch_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align weekly indicators to daily timeframe
    ema_34_1d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    atr_1w_1d = align_htf_to_ltf(prices, df_1w, atr_1w)
    donch_upper_1d = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_1d = align_htf_to_ltf(prices, df_1d, donch_lower)
    
    # Daily volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20  # Moderate volume surge
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d[i]) or np.isnan(atr_1w_1d[i]) or 
            np.isnan(donch_upper_1d[i]) or np.isnan(donch_lower_1d[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume surge and above weekly EMA34
            if (prices['close'].values[i] > donch_upper_1d[i] and vol_surge[i] and 
                prices['close'].values[i] > ema_34_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume surge and below weekly EMA34
            elif (prices['close'].values[i] < donch_lower_1d[i] and vol_surge[i] and 
                  prices['close'].values[i] < ema_34_1d[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses opposite Donchian level or volatility drops significantly
            if position == 1:
                if (prices['close'].values[i] < donch_lower_1d[i] or 
                    atr_1w_1d[i] < 0.5 * atr_1w_1d[i-1]):  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (prices['close'].values[i] > donch_upper_1d[i] or 
                    atr_1w_1d[i] < 0.5 * atr_1w_1d[i-1]):  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_WeeklyEMA34_VolumeSurge_v2"
timeframe = "1d"
leverage = 1.0