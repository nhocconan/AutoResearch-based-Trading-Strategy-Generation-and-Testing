#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout for trend direction and 1d Williams %R for mean reversion timing
# - Uses 4h HTF for Donchian channel (20-period): price above/below channel determines trend
# - Uses 1d HTF for Williams %R (14-period): extreme readings (<20 oversold, >80 overbought) for entry timing
# - In bullish 4h trend (price > upper Donchian): look for long entries when daily Williams %R < 20
# - In bearish 4h trend (price < lower Donchian): look for short entries when daily Williams %R > 80
# - Volume confirmation: current 1h volume > 1.5x 20-period average to avoid low-volume false signals
# - Session filter: only trade between 08:00-20:00 UTC to reduce noise
# - Fixed position size 0.20 to control drawdown and enable discrete sizing
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Uses discrete signal levels (0.0, ±0.20) to minimize fee churn from frequent small changes

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
    
    # Pre-compute session hours (08:00-20:00 UTC) - avoids datetime conversion in loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
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
    
    # Calculate 4h Donchian channel (20-period)
    period20_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period14_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    period14_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (period14_high - close_1d) / (period14_high - period14_low + 1e-10) * -100
    
    # Align all HTF data to 1h timeframe (wait for completed HTF bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Determine 4h trend: price above/below Donchian channel
        bullish_trend = close[i] > donchian_upper_aligned[i]
        bearish_trend = close[i] < donchian_lower_aligned[i]
        
        # Williams %R extremes: <20 = oversold, >80 = overbought
        oversold = williams_r_aligned[i] < 20
        overbought = williams_r_aligned[i] > 80
        
        # Fixed position size (discrete level for minimal fee churn)
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit conditions
            if bullish_trend:
                # In bullish 4h trend: exit when overbought or trend changes to bearish
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
                # In bearish 4h trend: exit when oversold or trend changes to bullish
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
            # Entry logic based on 4h trend and 1d Williams %R extremes
            if volume_confirmed:
                if bullish_trend and oversold:
                    # In bullish 4h trend, daily oversold: long mean reversion
                    position = 1
                    signals[i] = position_size
                elif bearish_trend and overbought:
                    # In bearish 4h trend, daily overbought: short mean reversion
                    position = -1
                    signals[i] = -position_size
    
    return signals