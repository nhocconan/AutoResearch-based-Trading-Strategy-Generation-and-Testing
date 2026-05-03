#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h EMA200 trend filter + volume spike confirmation.
# Williams %R identifies overbought/oversold conditions on 6h timeframe.
# In bull regime (price > 12h EMA200), look for mean reentry from oversold (%R < -80).
# In bear regime (price < 12h EMA200), look for mean reentry from overbought (%R > -20).
# Volume spike confirms participation. Designed for 50-150 total trades over 4 years.
# Works in both bull (buy dips) and bear (sell rallies) markets via regime filter.

name = "6h_WilliamsR_12hEMA200_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA200 for trend regime
    close_12h = df_12h['close'].values
    ema_200 = pd.Series(close_12h).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Calculate Williams %R on 6h (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        wr = williams_r[i]
        ema_trend = ema_200_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 12h EMA200, bear if close < 12h EMA200
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: mean reentry from oversold (Williams %R < -80) with volume spike
            long_entry = (wr < -80) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: mean reentry from overbought (Williams %R > -20) with volume spike
            short_entry = (wr > -20) and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on overbought condition (Williams %R > -20) or regime change to bear
            if wr > -20 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on oversold condition (Williams %R < -80) or regime change to bull
            if wr < -80 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals