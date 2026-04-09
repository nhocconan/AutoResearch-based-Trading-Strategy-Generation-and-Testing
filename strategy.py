#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout for direction and 1d Williams %R for mean reversion timing
# - Uses 4h HTF for Donchian channel (20-period) to establish trend: price above/below channel
# - Uses 1d HTF for Williams %R (14-period): extreme readings (<20 oversold, >80 overbought) for entry timing
# - In bullish 4h trend (price > upper Donchian): look for long entries when 1d Williams %R < 20 (pullback to oversold)
# - In bearish 4h trend (price < lower Donchian): look for short entries when 1d Williams %R > 80 (pullback to overbought)
# - Session filter: only trade between 08:00-20:00 UTC to avoid low-volume Asian session noise
# - Fixed position size 0.20 to control drawdown and enable discrete sizing
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Exit on opposite signal or when price reverts to mid-channel (mean reversion complete)

name = "1h_4h_1d_donchian_williams_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08:00-20:00 UTC) - prices.index is already DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channel (20 periods)
    period20_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1d Williams %R (14 periods)
    period14_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    period14_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (period14_high - close_1d) / (period14_high - period14_low + 1e-10) * -100
    
    # Align all HTF data to 1h timeframe (wait for completed HTF bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if outside trading session or any required data is invalid
        if not in_session[i] or \
           np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or \
           np.isnan(donchian_middle_aligned[i]) or np.isnan(williams_r_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend based on Donchian channel
        bullish_4h_trend = close[i] > donchian_upper_aligned[i]
        bearish_4h_trend = close[i] < donchian_lower_aligned[i]
        
        # Williams %R extremes: <20 = oversold, >80 = overbought
        oversold = williams_r_aligned[i] < 20
        overbought = williams_r_aligned[i] > 80
        
        # Fixed position size
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit conditions: trend change or mean reversion complete (price to middle)
            if bearish_4h_trend or close[i] >= donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: trend change or mean reversion complete (price to middle)
            if bullish_4h_trend or close[i] <= donchian_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: only trade in direction of 4h trend on 1d Williams %R extreme pullback
            if bullish_4h_trend and oversold:
                # In bullish 4h trend, 1d oversold: long the pullback
                position = 1
                signals[i] = position_size
            elif bearish_4h_trend and overbought:
                # In bearish 4h trend, 1d overbought: short the pullback
                position = -1
                signals[i] = -position_size
    
    return signals