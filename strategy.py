#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume confirmation.
# Long when Williams %R crosses above -50 in uptrend (1d EMA34 rising), short when crosses below -50 in downtrend.
# Volume > 1.3x 20-period average confirms momentum. EMA34 filters out weak trends and chop.
# Target: 20-40 trades/year by requiring trend alignment and momentum confirmation.
# Works in bull/bear: EMA34 trend filter ensures only traded in established trends, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend direction
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = ema_34 > np.roll(ema_34, 1)
    ema_34_rising[0] = False  # first period
    
    # Align EMA34 rising to 12h timeframe
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    
    # Calculate Williams %R(14) on 12h data
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Previous Williams %R for crossover detection
    prev_williams_r = np.roll(williams_r, 1)
    prev_williams_r[0] = williams_r[0]  # first period
    
    # Williams %R crosses above -50 (bullish momentum)
    williams_cross_up = (prev_williams_r <= -50) & (williams_r > -50)
    # Williams %R crosses below -50 (bearish momentum)
    williams_cross_down = (prev_williams_r >= -50) & (williams_r < -50)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema_34_rising_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend filter: EMA34 rising (uptrend) or falling (downtrend)
        ema_uptrend = ema_34_rising_aligned[i]
        ema_downtrend = ~ema_34_rising_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Williams %R crosses above -50 in uptrend
                if williams_cross_up[i] and ema_uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -50 in downtrend
                elif williams_cross_down[i] and ema_downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R crosses below -50 (loss of momentum) or trend reverses
                if williams_cross_down[i] or not ema_uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R crosses above -50 (loss of momentum) or trend reverses
                if williams_cross_up[i] or not ema_downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR14_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0