# 4h_Angle_of_Attack_V1
# Hypothesis: On 4h timeframe, measure the angle of price movement over 3 periods as a proxy for momentum strength. 
# Enter long when angle > 30 degrees (strong upward momentum) with volume confirmation and price above 200 EMA.
# Enter short when angle < -30 degrees (strong downward momentum) with volume confirmation and price below 200 EMA.
# Exit when angle returns to neutral range (-10 to 10 degrees) or volume drops.
# This captures strong momentum moves while avoiding chop. The 200 EMA filter ensures we only trade with the long-term trend.
# Volume confirmation ensures moves are supported by participation. Designed to work in both bull (catch rallies) and bear (catch crashes) markets.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from math import degrees, atan2
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load daily data for 200 EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4-period price change for angle calculation (3 intervals = 4 points)
    close = prices['close'].values
    # Price change over 3 periods (4 points: current and 3 periods ago)
    price_change = close - np.roll(close, 4)
    # Time constant: 3 periods * 4 hours = 12 hours in price units
    # We'll use a fixed time value since we're measuring angle in price-time space
    time_interval = 3  # 3 periods
    # Avoid division by zero
    angles = np.zeros_like(close)
    for i in range(4, n):
        if time_interval != 0:
            # Calculate angle in degrees: arctan(price_change / time_interval) * (180/pi)
            angles[i] = degrees(atan2(price_change[i], time_interval))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if EMA not ready
        if np.isnan(ema_200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price > EMA200 for long, price < EMA200 for short
        trend_long = price > ema_200_1d_aligned[i]
        trend_short = price < ema_200_1d_aligned[i]
        
        # Momentum conditions
        angle = angles[i]
        strong_up = angle > 30.0    # >30 degrees = strong upward momentum
        strong_down = angle < -30.0  # <-30 degrees = strong downward momentum
        neutral = abs(angle) <= 10.0  # -10 to 10 degrees = neutral/no strong momentum
        
        if position == 0:
            # Long: strong upward momentum + volume confirmation + uptrend
            if strong_up and volume_ok and trend_long:
                signals[i] = 0.25
                position = 1
            # Short: strong downward momentum + volume confirmation + downtrend
            elif strong_down and volume_ok and trend_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: momentum turns neutral or down OR trend turns bearish
            if neutral or strong_down or not trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: momentum turns neutral or up OR trend turns bullish
            if neutral or strong_up or not trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Angle_of_Attack_V1"
timeframe = "4h"
leverage = 1.0