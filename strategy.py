#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h Trend Filter + Volume Spike
# Williams %R: momentum oscillator measuring overbought/oversold levels
# Long when Williams %R crosses above -80 from below in uptrend (price > 12h EMA50)
# Short when Williams %R crosses below -20 from above in downtrend (price < 12h EMA50)
# Volume spike (>1.8x 20-period average) confirms conviction
# Works in bull/bear: 12h EMA50 filter ensures we trade with higher timeframe trend
# Target: 20-35 trades/year by requiring trend alignment + Williams %R reversal + volume

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Williams %R (14-period) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Previous values for crossover detection
    wr_prev = np.roll(wr, 1)
    wr_prev[0] = np.nan
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(wr[i]) or np.isnan(wr_prev[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirm = volume > 1.8 * vol_ma[i]
        
        # Trend filter: price vs 12h EMA50
        uptrend = price > ema50_12h_aligned[i]
        downtrend = price < ema50_12h_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Williams %R crosses above -80 from below in uptrend
                if wr[i] > -80 and wr_prev[i] <= -80 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above in downtrend
                elif wr[i] < -20 and wr_prev[i] >= -20 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Williams %R goes below -50 or trend fails
                if wr[i] < -50 or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Williams %R goes above -50 or trend fails
                if wr[i] > -50 or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_12hEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0