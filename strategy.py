#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R extreme readings with volume confirmation and ADX trend filter.
# Enter long when weekly Williams %R < -80 (oversold) with volume spike and ADX > 25 (trending).
# Enter short when weekly Williams %R > -20 (overbought) with volume spike and ADX > 25.
# Uses discrete position sizing (0.30) to balance return and drawdown. Target: 15-25 trades/year.
# Williams %R provides mean reversion edge in ranging markets, volume confirms participant interest,
# ADX filter ensures we trade in trending conditions where reversals are more likely to sustain.
# Works in bull (buy oversold dips) and bear (sell overbought rallies) markets.

name = "1d_WilliamsR_Extreme_Volume_ADXFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams %R (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Williams %R (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    n_1w = len(high_1w)
    williams_r = np.full(n_1w, np.nan)
    
    for i in range(14, n_1w):
        highest_high = np.max(high_1w[i-14:i+1])
        lowest_low = np.min(low_1w[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1w[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50.0
    
    # Forward fill Williams %R
    williams_r = pd.Series(williams_r).ffill().values
    
    # Calculate 1d ADX (14-period) for trend strength
    def calculate_adx(high, low, close, length=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=length, adjust=False, min_periods=length).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=length, adjust=False, min_periods=length).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=length, adjust=False, min_periods=length).mean().values / atr
        
        dx = np.zeros_like(close)
        for i in range(length, len(close)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
        
        adx = pd.Series(dx).ewm(span=length, adjust=False, min_periods=length).mean().values
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    adx_trending = adx > 25  # Strong trend when ADX > 25
    
    # Calculate 1d volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions with volume confirmation and ADX trend filter
        long_signal = williams_r[i] < -80 and volume_spike[i] and adx_trending[i]
        short_signal = williams_r[i] > -20 and volume_spike[i] and adx_trending[i]
        
        # Exit conditions: Williams %R returns to neutral territory
        long_exit = williams_r[i] > -50
        short_exit = williams_r[i] < -50
        
        # Handle entries and exits
        if long_signal and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.30
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals