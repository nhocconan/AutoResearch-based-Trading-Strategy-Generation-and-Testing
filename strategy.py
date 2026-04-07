#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray + 1d Trend + Volume Confirmation
# Hypothesis: Elder Ray (Bull/Bear Power) on 6h identifies institutional buying/selling pressure.
# In bull regime (1d EMA > 1d SMA), we long when Bull Power > 0 and rising.
# In bear regime (1d EMA < 1d SMA), we short when Bear Power < 0 and falling.
# Volume confirms institutional participation. Works in trending and mean-reverting markets.
# 6h timeframe balances responsiveness and noise. Target: 12-37 trades/year (50-150 over 4 years).
name = "6h_elder_ray_1d_trend_volume_v1"
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 13-period EMA for Elder Ray (standard)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # 1-day EMA(50) and SMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_sma50 = pd.Series(daily_close).rolling(window=50, min_periods=50).mean().values
    daily_ema50_6h = align_htf_to_ltf(prices, df_1d, daily_ema50)
    daily_sma50_6h = align_htf_to_ltf(prices, df_1d, daily_sma50)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(daily_ema50_6h[i]) or np.isnan(daily_sma50_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime from 1d data
        bull_regime = daily_ema50_6h[i] > daily_sma50_6h[i]  # Bullish when EMA > SMA
        bear_regime = daily_ema50_6h[i] < daily_sma50_6h[i]  # Bearish when EMA < SMA
        
        if position == 1:  # Long position
            # Exit: Bear Power turns negative with volume or trend reversal
            if bear_power[i] < 0 and vol_filter[i]:
                position = 0
                signals[i] = 0.0
            elif not bull_regime:  # Trend turned bearish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Bull Power turns positive with volume or trend reversal
            if bull_power[i] > 0 and vol_filter[i]:
                position = 0
                signals[i] = 0.0
            elif not bear_regime:  # Trend turned bullish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Bull regime: look for long when Bull Power > 0 and rising
                if bull_regime and bull_power[i] > 0:
                    # Check if Bull Power is rising (current > previous)
                    if i > 60 and bull_power[i] > bull_power[i-1]:
                        position = 1
                        signals[i] = 0.25
                # Bear regime: look for short when Bear Power < 0 and falling
                elif bear_regime and bear_power[i] < 0:
                    # Check if Bear Power is falling (current < previous)
                    if i > 60 and bear_power[i] < bear_power[i-1]:
                        position = -1
                        signals[i] = -0.25
    
    return signals