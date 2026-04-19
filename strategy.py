#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d/1w trend filter and volume confirmation.
# Long when: Close breaks above 12h Donchian upper band AND 1d EMA200 rising AND volume > 1.5x 20-period average
# Short when: Close breaks below 12h Donchian lower band AND 1d EMA200 falling AND volume > 1.5x 20-period average
# Exit: Price crosses 12h Donchian middle band OR trend reversal
# Uses 1d/1w multi-timeframe for trend filtering to avoid false breakouts in choppy markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).
name = "12h_Donchian_20_1dEMA200_1wTrend_Volume"
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
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1d EMA200 for trend filter (updated daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1d EMA200 slope for trend direction
    ema200_slope = np.diff(ema200_1d_aligned, prepend=ema200_1d_aligned[0])
    
    # 1w trend filter (weekly close > weekly open for uptrend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    weekly_bullish = df_1w['close'].values > df_1w['open'].values
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)  # Ensure Donchian and EMA200 are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Break above upper band + rising EMA200 + weekly bullish + volume spike
            if (price > donch_high[i] and 
                ema200_slope[i] > 0 and 
                weekly_bullish_aligned[i] > 0.5 and 
                vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below lower band + falling EMA200 + weekly bearish + volume spike
            elif (price < donch_low[i] and 
                  ema200_slope[i] < 0 and 
                  weekly_bullish_aligned[i] < 0.5 and 
                  vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below middle band OR trend turns bearish
            if price < donch_mid[i] or ema200_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above middle band OR trend turns bullish
            if price > donch_mid[i] or ema200_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals