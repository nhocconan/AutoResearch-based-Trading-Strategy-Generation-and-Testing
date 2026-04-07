#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h EMA(50) trend filter + volume confirmation
# Elder Ray calculates bull power (High - EMA) and bear power (Low - EMA) to measure bullish/bearish strength.
# Long when bull power > 0 and increasing, bear power < 0 and increasing, price > 12h EMA(50), and volume > 20-period average.
# Short when bear power < 0 and decreasing, bull power > 0 and decreasing, price < 12h EMA(50), and volume > 20-period average.
# Uses 12h EMA for trend filter to avoid counter-trend trades, targeting 50-150 total trades over 4 years.

name = "6h_elder_ray_12h_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h EMA for Elder Ray (same as trend filter)
    ema_12h = ema_50_12h_aligned  # Reuse 12h EMA(50) for Elder Ray calculation
    
    # Elder Ray components
    bull_power = high - ema_12h
    bear_power = low - ema_12h
    
    # Slope of bull/bear power (3-period change)
    bull_power_slope = pd.Series(bull_power).diff(3).values
    bear_power_slope = pd.Series(bear_power).diff(3).values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(bull_power_slope[i]) or 
            np.isnan(bear_power_slope[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 12h EMA
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0 and bull_power_slope[i] > 0  # Bull power positive and rising
        bear_weak = bear_power[i] < 0 and bear_power_slope[i] > 0   # Bear power negative but rising (less bearish)
        bear_strong = bear_power[i] < 0 and bear_power_slope[i] < 0  # Bear power negative and falling
        bull_weak = bull_power[i] > 0 and bull_power_slope[i] < 0   # Bull power positive but falling (less bullish)
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit when bull power weakens or trend reverses
            if not bull_strong or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit when bear power weakens or trend reverses
            if not bear_strong or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with trend and volume confirmation
            # Long when bull power strong and bear weakening in uptrend
            if bull_strong and bear_weak and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short when bear power strong and bull weakening in downtrend
            elif bear_strong and bull_weak and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals