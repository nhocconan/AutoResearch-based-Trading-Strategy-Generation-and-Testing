#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining weekly Bollinger Band mean reversion with volume confirmation and trend filter from 1-day EMA50.
# Uses weekly Bollinger Bands (20,2) to identify overextended price extremes, confirmed by volume spikes,
# and filtered by daily EMA50 trend to trade with the intermediate-term bias.
# Designed to work in both bull and bear markets by taking mean-reversion trades only when aligned
# with the daily trend, reducing false signals in choppy conditions.
# Target: 15-30 trades/year per disciplined entries with clear risk management.
name = "12h_EMA50_1d_Bollinger20_2_Weekly_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly Bollinger Bands (20, 2)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    sma_20_1w = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    upper_band_1w = sma_20_1w + (2 * std_20_1w)
    lower_band_1w = sma_20_1w - (2 * std_20_1w)
    
    upper_band_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_band_1w)
    lower_band_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_band_1w)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_band_1w_aligned[i]) or 
            np.isnan(lower_band_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches or breaks below weekly lower Bollinger Band, above daily EMA50, with volume spike
            if (low[i] <= lower_band_1w_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches or breaks above weekly upper Bollinger Band, below daily EMA50, with volume spike
            elif (high[i] >= upper_band_1w_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to weekly SMA (mean reversion complete) or breaks below daily EMA50
            if (close[i] >= sma_20_1w[-1] if len(sma_20_1w) > 0 else 0) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to weekly SMA (mean reversion complete) or breaks above daily EMA50
            if (close[i] <= sma_20_1w[-1] if len(sma_20_1w) > 0 else 0) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals