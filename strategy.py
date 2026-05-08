#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Bollinger Band breakout with weekly trend filter and volume confirmation
# Long when price breaks above upper Bollinger Band (20,2) with weekly uptrend and volume spike
# Short when price breaks below lower Bollinger Band (20,2) with weekly downtrend and volume spike
# Bollinger Bands identify volatility expansion; weekly trend filters for higher timeframe direction
# Volume spike confirms institutional participation; avoids false signals
# Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag

name = "1d_BollingerBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Bollinger Bands (20,2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(sma20[i]) or 
            np.isnan(std20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        close_price = close[i]
        upper = upper_band[i]
        lower = lower_band[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper BB, weekly uptrend, volume spike
            if close_price > upper and ema50_1w_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower BB, weekly downtrend, volume spike
            elif close_price < lower and ema50_1w_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below middle Bollinger Band (SMA20) or weekly trend turns down
            if close_price < sma20[i] or ema50_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above middle Bollinger Band (SMA20) or weekly trend turns up
            if close_price > sma20[i] or ema50_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals