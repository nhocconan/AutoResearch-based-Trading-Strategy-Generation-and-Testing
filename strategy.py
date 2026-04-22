#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly trend filter and volume confirmation
# Weekly trend (price > 50-week EMA) provides bias for breakouts, reducing false signals
# Volume surge confirms institutional participation
# Designed for 12-37 trades/year (50-150 over 4 years) with 0.25 position sizing
# Works in bull markets via breakouts, in bear via short breakdowns with trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Weekly data for trend filter (50-period EMA)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # 6h ATR for volatility filter (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day Donchian high with volume surge and weekly uptrend
            if close[i] > donch_high_20_aligned[i] and vol_surge[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day Donchian low with volume surge and weekly downtrend
            elif close[i] < donch_low_20_aligned[i] and vol_surge[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or volatility drops significantly
            if position == 1:
                if close[i] < donch_low_20_aligned[i] or atr[i] < 0.5 * atr[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donch_high_20_aligned[i] or atr[i] < 0.5 * atr[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyTrend_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0