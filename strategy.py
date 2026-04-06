#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and 1d ATR filter
# Long when price breaks above 12h Donchian(20) high + volume > 1.5x avg + 1d ATR > 10-day median ATR
# Short when price breaks below 12h Donchian(20) low + volume > 1.5x avg + 1d ATR > 10-day median ATR
# Exit when price crosses Donchian midline or ATR drops below threshold
# Targets 50-150 trades over 4 years by requiring volatility expansion + volume confirmation

name = "12h_donchian_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    # 1d ATR filter - only trade when volatility is elevated
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate True Range for daily
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr_daily = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) on daily
    atr_daily = pd.Series(tr_daily).rolling(window=10, min_periods=10).mean().values
    # Median ATR over 50 days for dynamic threshold
    median_atr = pd.Series(atr_daily).rolling(window=50, min_periods=50).median().values
    atr_threshold = median_atr  # Trade only when current ATR >= median ATR
    
    # Align daily ATR to 12h timeframe
    atr_daily_aligned = align_htf_to_ltf(prices, df_1d, atr_daily)
    median_atr_aligned = align_htf_to_ltf(prices, df_1d, median_atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(atr_daily_aligned[i]) or 
            np.isnan(median_atr_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or atr_daily_aligned[i] < median_atr_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or atr_daily_aligned[i] < median_atr_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation and volatility filter
            # Long breakout: price above Donchian high + volume spike + elevated volatility
            if (close[i] > donchian_high[i-1] and 
                volume[i] > volume_threshold[i] and 
                atr_daily_aligned[i] >= median_atr_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price below Donchian low + volume spike + elevated volatility
            elif (close[i] < donchian_low[i-1] and 
                  volume[i] > volume_threshold[i] and 
                  atr_daily_aligned[i] >= median_atr_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals