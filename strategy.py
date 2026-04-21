#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 in uptrend (1d EMA34 rising), short when crosses below -20 in downtrend.
# Volume > 1.5x 20-period average confirms reversal strength. Uses EMA slope to filter weak trends.
# Target: 20-40 trades/year by requiring strong momentum + volume + trend alignment.
# Works in bull/bear: EMA filter ensures only established trends are traded, avoiding whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate EMA slope for trend strength
    ema_slope = np.diff(ema_34, prepend=ema_34[0])
    
    # Align EMA and slope to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    # Calculate Williams %R on 12h data (14-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_slope_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: EMA slope positive for uptrend, negative for downtrend
        uptrend = ema_slope_aligned[i] > 0
        downtrend = ema_slope_aligned[i] < 0
        
        if position == 0:
            if volume_confirm:
                # Long: Williams %R crosses above -80 in uptrend
                if williams_r[i] > -80 and williams_r[i-1] <= -80 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 in downtrend
                elif williams_r[i] < -20 and williams_r[i-1] >= -20 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R rises above -20 (overbought) or trend changes
                if williams_r[i] > -20 or ema_slope_aligned[i] < 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R falls below -80 (oversold) or trend changes
                if williams_r[i] < -80 or ema_slope_aligned[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR14_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0