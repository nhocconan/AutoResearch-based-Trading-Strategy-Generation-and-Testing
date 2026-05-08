#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Bollinger Band breakout with weekly trend filter and volume confirmation
# Designed for low trade frequency on daily timeframe: 10-25 trades/year.
# Long when price breaks above upper BB(20,2) with weekly EMA(34) uptrend and volume spike.
# Short when price breaks below lower BB(20,2) with weekly EMA(34) downtrend and volume spike.
# Uses mean reversion exits when price returns to BB middle band.
# Target: 20-80 total trades over 4 years = 5-20/year

name = "1d_BollingerBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Bollinger Bands on daily data
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    middle_band = sma
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 34)  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(sma[i]) or 
            np.isnan(std[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        upper = upper_band[i]
        lower = lower_band[i]
        middle = middle_band[i]
        vol_spike = volume_spike[i]
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB + weekly uptrend + volume spike
            if (price > upper and 
                ema34_1w_val > sma[i] and  # weekly EMA > daily SMA = uptrend
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB + weekly downtrend + volume spike
            elif (price < lower and 
                  ema34_1w_val < sma[i] and  # weekly EMA < daily SMA = downtrend
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB
            if price <= middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB
            if price >= middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals