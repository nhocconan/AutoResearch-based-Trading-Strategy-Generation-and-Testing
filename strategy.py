#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with weekly trend filter (EMA34) and volume confirmation.
# Donchian breakouts capture momentum in trending markets, while weekly EMA34 filters for trend direction.
# Volume confirmation ensures breakouts are supported by participation, reducing false signals.
# Designed for 1d timeframe to capture multi-day trends with low frequency (<15 trades/year).
# Entry: Long when price > 20-day high and price > weekly EMA34 and volume spike.
#        Short when price < 20-day low and price < weekly EMA34 and volume spike.
# Exit: Opposite Donchian break (10-day) or trend reversal (price crosses weekly EMA34).
# Uses strict conditions to limit trades (~10-20/year) and avoid overtrading.
name = "1d_Donchian20_WeeklyEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 20-day high with uptrend and volume
            if (close[i] > donchian_high[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low with downtrend and volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks 10-day low or crosses below weekly EMA34
            if (close[i] < donchian_low[i]) or (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks 10-day high or crosses above weekly EMA34
            if (close[i] > donchian_high[i]) or (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals