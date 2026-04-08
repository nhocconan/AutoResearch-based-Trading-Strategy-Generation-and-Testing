#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray Power + Weekly Trend + Volume Confirmation
# Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures
# bull/bear strength relative to trend. Combined with weekly trend (EMA50 > SMA50 = bull),
# we go long when Bull Power > 0 and rising in bull regime, short when Bear Power > 0
# and rising in bear regime. Volume confirms institutional participation.
# 6h timeframe balances responsiveness and noise. Target: 12-37 trades/year (50-150 over 4 years).
name = "6h_elder_ray_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Weekly trend filter: EMA50 > SMA50 = bullish
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_ema50_6h = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    weekly_sma50_6h = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(weekly_ema50_6h[i]) or np.isnan(weekly_sma50_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime from weekly data
        bull_regime = weekly_ema50_6h[i] > weekly_sma50_6h[i]  # Bullish when EMA > SMA
        bear_regime = weekly_ema50_6h[i] < weekly_sma50_6h[i]  # Bearish when EMA < SMA
        
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 (weakening bullish momentum) or trend reversal
            if bull_power[i] <= 0 or not bull_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Bear Power <= 0 (weakening bearish momentum) or trend reversal
            if bear_power[i] <= 0 or not bear_regime:
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
                    if i > 30 and bull_power[i] > bull_power[i-1]:
                        position = 1
                        signals[i] = 0.25
                # Bear regime: look for short when Bear Power > 0 and rising
                elif bear_regime and bear_power[i] > 0:
                    # Check if Bear Power is rising (current > previous)
                    if i > 30 and bear_power[i] > bear_power[i-1]:
                        position = -1
                        signals[i] = -0.25
    
    return signals