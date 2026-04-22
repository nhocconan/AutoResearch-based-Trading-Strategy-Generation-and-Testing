#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h range breakout using 4h Keltner bands with volume confirmation and 1d trend filter
# Uses 4h ATR-based channels to capture volatility expansion and 1d EMA50 for trend alignment
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) with strict entry conditions
# Breakouts occur when price breaks 4h Keltner bands with volume spike, filtered by 1d trend
# Works in bull/bear via trend filter; session filter reduces noise

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4-hour data for Keltner channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ATR(20) for Keltner channels
    tr_4h = np.maximum(np.maximum(high_4h[1:] - low_4h[1:], 
                                  np.abs(high_4h[1:] - close_4h[:-1])),
                       np.abs(low_4h[1:] - close_4h[:-1]))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h EMA20 for Keltner center
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner upper and lower bands (2.0 * ATR)
    keltner_upper = ema_20_4h + 2.0 * atr_4h
    keltner_lower = ema_20_4h - 2.0 * atr_4h
    
    # Load 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 1h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 1-hour timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_4h, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_4h, keltner_lower)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Keltner band + volume spike + uptrend (price > 1d EMA50)
            if (close[i] > keltner_upper_aligned[i] and vol_spike[i] and close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below lower Keltner band + volume spike + downtrend (price < 1d EMA50)
            elif (close[i] < keltner_lower_aligned[i] and vol_spike[i] and close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Price returns to 4h EMA20 (middle of Keltner channel)
            if position == 1:
                if close[i] < ema_20_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > ema_20_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Keltner_Breakout_Volume_Trend_Session"
timeframe = "1h"
leverage = 1.0