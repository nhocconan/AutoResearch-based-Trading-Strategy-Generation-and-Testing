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
    
    # Load 1-day data for ATR and close
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(7) and ATR(30) on daily timeframe
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # First TR
    
    atr7 = pd.Series(tr1).rolling(window=7, min_periods=7).mean().values
    atr30 = pd.Series(tr1).rolling(window=30, min_periods=30).mean().values
    
    # ATR ratio: ATR(7)/ATR(30) > 2.0 indicates volatility expansion
    atr_ratio = np.where(atr30 > 0, atr7 / atr30, 0)
    
    # Load 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA(50) for trend direction
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    
    # Bollinger Bands (20, 2.5) on daily for mean reversion signal
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    lower_band = sma20 - 2.5 * std20
    
    # Align all indicators to 6-hour timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    sma20_aligned = align_htf_to_ltf(prices, df_1d, sma20)
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(lower_band_aligned[i]) or np.isnan(sma20_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Volatility expansion + price at/below lower BB + weekly uptrend + volume spike
            if (atr_ratio_aligned[i] > 2.0 and 
                close[i] <= lower_band_aligned[i] and 
                weekly_uptrend_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
        else:
            # Exit: Price returns to SMA(20) or volatility contracts
            if position == 1:
                if (close[i] >= sma20_aligned[i] or atr_ratio_aligned[i] < 1.2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
    
    return signals

name = "6h_ATR_Vol_Spike_BB_MeanReversion_WeeklyTrend"
timeframe = "6h"
leverage = 1.0