#/usr/bin/env python3
# Hypothesis: 1d Bollinger Band breakout with weekly trend filter and volume confirmation
# Uses daily close to detect Bollinger Band breakouts (20,2). 
# Long when: close > upper band, weekly EMA(50) rising, volume spike (>1.5x 20-day average)
# Short when: close < lower band, weekly EMA(50) falling, volume spike
# Exit when: price crosses the middle band (20-day SMA) OR trend reverses
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 10-25 trades/year.
# Designed to work in both bull (breakouts) and bear (mean-reversion at extremes) markets.

name = "1d_Bollinger_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20,2) on daily data
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    middle_band = sma_20  # 20-day SMA for exit
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close']
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_prev = np.roll(ema_50_1w, 1)
    ema_50_1w_prev[0] = ema_50_1w[0]
    ema_rising = ema_50_1w > ema_50_1w_prev
    ema_falling = ema_50_1w < ema_50_1w_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(middle_band[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: close > upper band + weekly EMA rising + volume spike
            if (close[i] > upper_band[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: close < lower band + weekly EMA falling + volume spike
            elif (close[i] < lower_band[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle band OR trend turns down
            if (close[i] < middle_band[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle band OR trend turns up
            if (close[i] > middle_band[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals