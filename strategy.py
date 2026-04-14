#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with weekly trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
# Weekly trend filter ensures alignment with higher timeframe trend
# Volume > 1.3x average confirms momentum
# Target: 15-25 trades/year per symbol to avoid fee drag
# Works in bull/bear markets by following the dominant trend

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Williams Alligator components (13,8,5 smoothed with 8,5,3)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(weekly_ema50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA50
        weekly_uptrend = close[i] > weekly_ema50_aligned[i]
        weekly_downtrend = close[i] < weekly_ema50_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Williams Alligator signals
        # Bullish alignment: Lips > Teeth > Jaw (all rising)
        bullish_aligned = (lips[i] > teeth[i] > jaw[i]) and \
                         (lips[i] > lips[i-1]) and \
                         (teeth[i] > teeth[i-1]) and \
                         (jaw[i] > jaw[i-1])
        
        # Bearish alignment: Jaw > Teeth > Lips (all falling)
        bearish_aligned = (jaw[i] > teeth[i] > lips[i]) and \
                         (jaw[i] < jaw[i-1]) and \
                         (teeth[i] < teeth[i-1]) and \
                         (lips[i] < lips[i-1])
        
        if position == 0:
            # Enter long: bullish alignment + weekly uptrend + volume
            if bullish_aligned and weekly_uptrend and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: bearish alignment + weekly downtrend + volume
            elif bearish_aligned and weekly_downtrend and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment or loss of weekly uptrend
            if bearish_aligned or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish alignment or loss of weekly downtrend
            if bullish_aligned or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WilliamsAlligator_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0