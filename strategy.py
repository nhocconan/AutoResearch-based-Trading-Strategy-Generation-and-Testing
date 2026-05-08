#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band width expansion breakout with 1-day trend filter and volume confirmation
# Long when price breaks above upper BB during BB width expansion + daily EMA(50) uptrend + volume spike
# Short when price breaks below lower BB during BB width expansion + daily EMA(50) downtrend + volume spike
# Bollinger Band width expansion indicates increasing volatility and potential trend start
# Daily trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation
# Targets 75-200 total trades over 4 years (19-50/year) to avoid fee drag

name = "4h_BBWidthExpansion_Breakout_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Bollinger Band width
    bb_width = (upper_band - lower_band) / sma
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_width_expanding = bb_width > bb_width_ma  # Width expanding above average
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(bb_width_expanding[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        upper = upper_band[i]
        lower = lower_band[i]
        width_expanding = bb_width_expanding[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB + width expanding + daily uptrend + volume spike
            if close[i] > upper and width_expanding and close[i] > ema50_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB + width expanding + daily downtrend + volume spike
            elif close[i] < lower and width_expanding and close[i] < ema50_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below middle BB OR daily trend turns down
            if close[i] < sma[i] or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above middle BB OR daily trend turns up
            if close[i] > sma[i] or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals