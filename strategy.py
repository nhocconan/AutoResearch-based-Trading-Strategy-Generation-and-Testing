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
    
    # Weekly 20-period Donchian channels (long-term trend)
    df_1w = get_htf_data(prices, '1w')
    donch_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    tr_1d = np.maximum(df_1d['high'] - df_1d['low'],
                       np.maximum(np.abs(df_1d['high'] - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]])),
                                  np.abs(df_1d['low'] - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: current > 1.5x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    # ADX filter for trending markets (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])), np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_filter = adx > 25  # Trending market
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_threshold[i]) or 
            np.isnan(adx[i])):
            continue
        
        # Long: price breaks above weekly Donchian high + volume + ADX filter
        if close[i] > donch_high_aligned[i] and volume[i] > vol_threshold[i] and adx_filter[i]:
            signals[i] = 0.25
        
        # Short: price breaks below weekly Donchian low + volume + ADX filter
        elif close[i] < donch_low_aligned[i] and volume[i] > vol_threshold[i] and adx_filter[i]:
            signals[i] = -0.25
        
        # Exit: price crosses back inside weekly Donchian channels (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_high_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > donch_low_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0