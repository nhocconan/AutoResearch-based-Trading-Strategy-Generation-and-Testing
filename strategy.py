#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-day high/low) for trend
    df_1d = get_htf_data(prices, '1d')
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # 4h RSI for momentum confirmation
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume filter: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(rsi[i])):
            continue
        
        # Long: price > daily Donchian high + RSI > 50 + volume confirmation
        if close[i] > donch_high_aligned[i] and rsi[i] > 50 and volume[i] > vol_threshold[i]:
            signals[i] = 0.25
        
        # Short: price < daily Donchian low + RSI < 50 + volume confirmation
        elif close[i] < donch_low_aligned[i] and rsi[i] < 50 and volume[i] > vol_threshold[i]:
            signals[i] = -0.25
        
        # Exit: price crosses back inside daily Donchian channels
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_high_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > donch_low_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyDonchian_RSI_Volume"
timeframe = "4h"
leverage = 1.0