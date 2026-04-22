#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike
# Uses Donchian channel breakouts for entries with daily trend filter to avoid counter-trend trades
# Long when price breaks above 20-period high with 1d uptrend and volume spike
# Short when price breaks below 20-period low with 1d downtrend and volume spike
# Daily trend filter provides stronger trend bias, reducing whipsaws in choppy markets
# Designed for 12h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years)
# Volume spike confirms institutional participation in breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and Donchian calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_upper_12h = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_12h = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # 1d EMA(50) for higher timeframe trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter (20-period on 12h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + 1d uptrend + volume spike
            if (close[i] > donchian_upper_12h[i] and 
                close[i] > ema_50_12h[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + 1d downtrend + volume spike
            elif (close[i] < donchian_lower_12h[i] and 
                  close[i] < ema_50_12h[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Donchian middle or trend reversal
            donchian_middle = (donchian_upper_12h[i] + donchian_lower_12h[i]) / 2
            
            if position == 1:
                # Exit on price below Donchian middle or trend reversal
                if (close[i] < donchian_middle or 
                    close[i] < ema_50_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above Donchian middle or trend reversal
                if (close[i] > donchian_middle or 
                    close[i] > ema_50_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0