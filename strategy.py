#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian breakout with weekly volume confirmation and monthly trend filter
# Long when price breaks above 1d Donchian upper band with volume spike and monthly bullish trend
# Short when price breaks below 1d Donchian lower band with volume spike and monthly bearish trend
# Exit when price crosses the Donchian midline
# Uses monthly EMA trend filter to avoid counter-trend trades in bear markets
# Target: 15-30 trades per symbol over 4 years (4-7.5/year) to minimize fee drag
# This strategy avoids overtrading by using higher timeframe (1d) and strict volume/spike conditions

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and monthly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_monthly = get_htf_data(prices, '1M')
    
    # Calculate 1d Donchian channel (20-period lookback for 1d timeframe)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate monthly EMA for trend filter (21-period)
    close_monthly = df_monthly['close'].values
    ema_monthly = pd.Series(close_monthly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1d timeframe (already aligned via daily data)
    donchian_upper_aligned = donchian_upper  # already daily
    donchian_lower_aligned = donchian_lower
    donchian_middle_aligned = donchian_middle
    ema_monthly_aligned = align_htf_to_ltf(prices, df_monthly, ema_monthly)
    vol_ma_1d_aligned = vol_ma_1d  # already daily
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_monthly_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = volume[i]  # Current 1d volume
        
        if position == 0:
            # Long setup: break above Donchian upper with volume spike and monthly bullish trend
            if (price > donchian_upper_aligned[i] and 
                vol_1d_current > 2.0 * vol_ma_1d_aligned[i] and  # Volume spike (stricter)
                price > ema_monthly_aligned[i]):                    # Price above monthly EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian lower with volume spike and monthly bearish trend
            elif (price < donchian_lower_aligned[i] and 
                  vol_1d_current > 2.0 * vol_ma_1d_aligned[i] and  # Volume spike
                  price < ema_monthly_aligned[i]):                    # Price below monthly EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian middle
            if price < donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian middle
            if price > donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_MonthlyTrend_Volume"
timeframe = "1d"
leverage = 1.0