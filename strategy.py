#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data (HTF) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 60-period EMA on 1d for trend filter (faster than 200)
    close_1d = df_1d['close'].values
    ema_60_1d = pd.Series(close_1d).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Calculate weekly RSI(14) for momentum filter
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1w = 100 - (100 / (1 + rs)).values
    
    # Calculate 6h Donchian channel (20-period)
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    ema_60_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_60_1d)
    rsi_14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_14_1w)
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)  # 1d Donchian on 6s
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_60_1d_aligned[i]) or np.isnan(rsi_14_1w_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 1d Donchian high with volume AND above daily EMA60 AND weekly RSI > 50
            if (close[i] > donch_high_20_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_aligned[i] and 
                close[i] > ema_60_1d_aligned[i] and 
                rsi_14_1w_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1d Donchian low with volume AND below daily EMA60 AND weekly RSI < 50
            elif (close[i] < donch_low_20_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_aligned[i] and 
                  close[i] < ema_60_1d_aligned[i] and 
                  rsi_14_1w_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite 1d Donchian level OR weekly RSI reaches extreme
            if position == 1:
                if (close[i] < donch_low_20_aligned[i] or 
                    rsi_14_1w_aligned[i] > 70):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > donch_high_20_aligned[i] or 
                    rsi_14_1w_aligned[i] < 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Donchian1D_EMA60_RSI1W_Trend_Momentum"
timeframe = "6h"
leverage = 1.0