#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band breakout with volume confirmation and daily trend filter.
# In bull markets, buy breakouts above upper band; in bear markets, sell breakouts below lower band.
# Uses daily EMA34 as trend filter: only take long trades when price > daily EMA34, short when price < daily EMA34.
# Bollinger Bands (20,2) provide dynamic support/resistance. Volume filter ensures breakouts have conviction.
# Target: 20-50 trades per year (80-200 total over 4 years) to avoid excessive fee drag.
name = "4h_BollingerBreakout_Volume_DailyTrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Bollinger Bands (20, 2) on 4h data
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Calculate daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        ema_trend = ema_34_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above upper band with volume confirmation and bullish daily trend
            if close_val > upper and vol_filter and (close_val > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume confirmation and bearish daily trend
            elif close_val < lower and vol_filter and (close_val < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below middle band (SMA20)
            if close_val < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above middle band (SMA20)
            if close_val > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals