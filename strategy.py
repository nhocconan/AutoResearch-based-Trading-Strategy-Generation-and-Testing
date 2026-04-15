#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop (for 12h primary timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period) for trend structure
    weekly_highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ATR(14) for volatility filter
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    weekly_tr = np.maximum(weekly_high - weekly_low,
                           np.maximum(np.abs(weekly_high - weekly_close_prev),
                                      np.abs(weekly_low - weekly_close_prev)))
    weekly_atr_14 = pd.Series(weekly_tr).rolling(window=14, min_periods=14).mean().values
    weekly_atr_ma_50 = pd.Series(weekly_atr_14).rolling(window=50, min_periods=50).mean().values
    weekly_volatility_ratio = weekly_atr_14 / (weekly_atr_ma_50 + 1e-10)
    
    # Calculate weekly RSI(14) for momentum filter
    weekly_delta = np.diff(weekly_close, prepend=weekly_close[0])
    weekly_gain = np.where(weekly_delta > 0, weekly_delta, 0)
    weekly_loss = np.where(weekly_delta < 0, -weekly_delta, 0)
    weekly_avg_gain = pd.Series(weekly_gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    weekly_avg_loss = pd.Series(weekly_loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    weekly_rs = weekly_avg_gain / (weekly_avg_loss + 1e-10)
    weekly_rsi_14 = 100 - (100 / (1 + weekly_rs))
    
    # Align HTF indicators to 12h timeframe with proper delay
    weekly_highest_20_12h = align_htf_to_ltf(prices, df_1w, weekly_highest_20)
    weekly_lowest_20_12h = align_htf_to_ltf(prices, df_1w, weekly_lowest_20)
    weekly_volatility_ratio_12h = align_htf_to_ltf(prices, df_1w, weekly_volatility_ratio)
    weekly_rsi_14_12h = align_htf_to_ltf(prices, df_1w, weekly_rsi_14)
    
    # Calculate 12h Donchian breakout levels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume confirmation (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_highest_20_12h[i]) or np.isnan(weekly_lowest_20_12h[i]) or
            np.isnan(weekly_volatility_ratio_12h[i]) or np.isnan(weekly_rsi_14_12h[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price relative to weekly Donchian channels
        # 2. Weekly momentum filter: RSI not extreme (avoid exhaustion)
        # 3. Volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 4. 12h Donchian breakout with volume confirmation
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: price breaks above weekly Donchian high with volume
        if (close[i] > weekly_highest_20_12h[i] and      # Above weekly Donchian high (bullish structure)
            weekly_rsi_14_12h[i] < 70 and               # Not overbought on weekly
            weekly_volatility_ratio_12h[i] > 0.8 and    # Avoid low volatility squeezes
            close[i] > highest_20[i] and                 # 12h Donchian breakout
            volume_ratio[i] > 1.5):                      # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: price breaks below weekly Donchian low with volume
        elif (close[i] < weekly_lowest_20_12h[i] and     # Below weekly Donchian low (bearish structure)
              weekly_rsi_14_12h[i] > 30 and              # Not oversold on weekly
              weekly_volatility_ratio_12h[i] > 0.8 and   # Avoid low volatility squeezes
              close[i] < lowest_20[i] and                 # 12h Donchian breakdown
              volume_ratio[i] > 1.5):                     # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WeeklyDonchianTrend_Volume_Breakout"
timeframe = "12h"
leverage = 1.0