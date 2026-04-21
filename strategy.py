#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d William's %R(14) mean reversion with 1w EMA(34) trend filter and volume spike confirmation.
# Long when %R crosses above -80 from below in uptrend (weekly EMA rising), short when %R crosses below -20 from above in downtrend.
# Volume > 1.3x 20-period average confirms momentum. EMA filter ensures trading with higher timeframe trend.
# Target: 10-25 trades/year by requiring oversold/overbought extremes + trend alignment + volume confirmation.
# Works in bull/bear: EMA filter adapts to trend direction, %R captures mean reversion within trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_rising = ema_1w > np.roll(ema_1w, 1)  # rising if current > previous
    ema_1w_falling = ema_1w < np.roll(ema_1w, 1)  # falling if current < previous
    
    # Align EMA trend signals to daily timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_falling)
    
    # Calculate Williams %R(14) on daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: 0 to -100, where above -20 is overbought, below -80 is oversold
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(wr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below in uptrend
            if (wr[i] > -80 and wr[i-1] <= -80 and 
                ema_rising_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above in downtrend
            elif (wr[i] < -20 and wr[i-1] >= -20 and 
                  ema_falling_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if %R crosses below -50 (mean reversion) or trend changes
                if wr[i] < -50 or not ema_rising_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if %R crosses above -50 (mean reversion) or trend changes
                if wr[i] > -50 or not ema_falling_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR14_EMA34Trend_Volume"
timeframe = "1d"
leverage = 1.0