#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout for trend direction and 1d Williams %R for mean reversion
# - Uses 1d HTF for Donchian channel: price above/below 20-period high/low determines trend
# - Uses 1d HTF for Williams %R: extreme readings (>80 or <20) signal mean reversion opportunities
# - In bullish trend (price > 20d high): look for long entries when daily Williams %R < 20 (oversold)
# - In bearish trend (price < 20d low): look for short entries when daily Williams %R > 80 (overbought)
# - Volume confirmation: current 12h volume > 1.3x 20-period average to avoid low-volume false signals
# - Fixed position size 0.25 to control drawdown
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_donchian_williams_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20 periods)
    period20_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Williams %R (14 periods)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period14_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    period14_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (period14_high - close_1d) / (period14_high - period14_low + 1e-10) * -100
    
    # Align all HTF data to 12h timeframe (wait for completed HTF bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, period20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, period20_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute volume confirmation (20-period average for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Trend determination: price above/below Donchian channel
        bullish_trend = close[i] > donchian_high_aligned[i]
        bearish_trend = close[i] < donchian_low_aligned[i]
        
        # Williams %R extremes: <20 = oversold, >80 = overbought
        oversold = williams_r_aligned[i] < 20
        overbought = williams_r_aligned[i] > 80
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if bullish_trend:
                # In bullish trend: exit when overbought or trend changes to bearish
                if overbought or bearish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Not in bullish trend: exit
                position = 0
                signals[i] = 0.0
                
        elif position == -1:  # Short position
            # Exit conditions
            if bearish_trend:
                # In bearish trend: exit when oversold or trend changes to bullish
                if oversold or bullish_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Not in bearish trend: exit
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on trend and Williams %R extremes
            if volume_confirmed:
                if bullish_trend and oversold:
                    # In bullish trend, daily oversold: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif bearish_trend and overbought:
                    # In bearish trend, daily overbought: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals