#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Williams %R + 1d Trend + Volume Confirmation
# Hypothesis: Williams %R identifies overbought/oversold conditions. In bull regime (1d EMA > 1d SMA),
# we go long when Williams %R crosses above -50 from below. In bear regime (1d EMA < 1d SMA),
# we go short when Williams %R crosses below -50 from above. Volume confirms institutional participation.
# Williams %R is mean-reverting but works with trend filter to avoid whipsaws.
# 4h timeframe balances responsiveness and noise. Target: 20-50 trades/year (80-200 over 4 years).
name = "4h_williams_r_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
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
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1-day EMA(50) and SMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_sma50 = pd.Series(daily_close).rolling(window=50, min_periods=50).mean().values
    daily_ema50_4h = align_htf_to_ltf(prices, df_1d, daily_ema50)
    daily_sma50_4h = align_htf_to_ltf(prices, df_1d, daily_sma50)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(daily_ema50_4h[i]) or 
            np.isnan(daily_sma50_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime from 1d data
        bull_regime = daily_ema50_4h[i] > daily_sma50_4h[i]  # Bullish when EMA > SMA
        bear_regime = daily_ema50_4h[i] < daily_sma50_4h[i]  # Bearish when EMA < SMA
        
        if position == 1:  # Long position
            # Exit: Williams %R goes below -50 (overbought) with volume or trend reversal
            if williams_r[i] < -50 and vol_filter[i]:
                position = 0
                signals[i] = 0.0
            elif not bull_regime:  # Trend turned bearish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Williams %R goes above -50 (oversold) with volume or trend reversal
            if williams_r[i] > -50 and vol_filter[i]:
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
                # Bull regime: look for long when Williams %R crosses above -50 from below
                if bull_regime and williams_r[i] > -50:
                    # Check if crossed above -50 (previous <= -50 and current > -50)
                    if i > 20 and williams_r[i-1] <= -50 and williams_r[i] > -50:
                        position = 1
                        signals[i] = 0.25
                # Bear regime: look for short when Williams %R crosses below -50 from above
                elif bear_regime and williams_r[i] < -50:
                    # Check if crossed below -50 (previous >= -50 and current < -50)
                    if i > 20 and williams_r[i-1] >= -50 and williams_r[i] < -50:
                        position = -1
                        signals[i] = -0.25
    
    return signals