#!/usr/bin/env python3
# Hypothesis: 1d price action trading with weekly trend filter
# Long when price closes above weekly high with RSI(14) > 50 and volume > 1.5x average
# Short when price closes below weekly low with RSI(14) < 50 and volume > 1.5x average
# Exit on opposite weekly extreme touch or RSI reversal
# Uses weekly trend for direction, daily price action for entry, volume for confirmation
# Designed for low frequency (target: 50-100 trades over 4 years) to minimize fee drag
# Works in bull markets via breakouts and bear via mean reversion at extremes

name = "1d_WeeklyTrend_DailyPriceAction_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly high/low for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly high and low (using close of completed weekly bar)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly levels to daily timeframe (available after weekly bar closes)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Daily RSI(14) for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # neutral when not enough data
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for RSI calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(rsi_values[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price closes above weekly high, RSI > 50, volume spike
            if (close[i] > weekly_high_aligned[i] and 
                rsi_values[i] > 50 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price closes below weekly low, RSI < 50, volume spike
            elif (close[i] < weekly_low_aligned[i] and 
                  rsi_values[i] < 50 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches weekly low or RSI turns bearish
            if (close[i] <= weekly_low_aligned[i]) or (rsi_values[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches weekly high or RSI turns bullish
            if (close[i] >= weekly_high_aligned[i]) or (rsi_values[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals