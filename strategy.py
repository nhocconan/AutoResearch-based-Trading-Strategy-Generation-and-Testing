#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d EMA200 trend filter and volume spike confirmation.
# Williams %R measures overbought/oversold levels. In bull regime (price > 1d EMA200),
# we go long when Williams %R crosses above -80 from below with volume spike.
# In bear regime (price < 1d EMA200), we go short when Williams %R crosses below -20 from above with volume spike.
# This captures mean reversion within the trend, works in both bull and bear markets.
# Target: 20-40 trades/year on 4h (80-160 total over 4 years) to minimize fee drag.

name = "4h_WilliamsR_1dTrend_VolumeSpike"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 trend filter
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate Williams %R (14-period) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume regime: current 4h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        wr = williams_r[i]
        ema_trend = ema_200_aligned[i]
        vol_spike = volume_spike[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if np.isnan(wr) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA200, bear if close < 1d EMA200
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Williams %R levels
        oversold = -80
        overbought = -20
        
        # Generate signals
        if position == 0:
            # Look for crossovers with volume spike
            if i > 100:
                wr_prev = williams_r[i-1]
                # Bullish crossover: WR crosses above -80 from below
                bull_cross = (wr_prev < oversold and wr >= oversold) and vol_spike
                # Bearish crossover: WR crosses below -20 from above
                bear_cross = (wr_prev > overbought and wr <= overbought) and vol_spike
                
                if is_bull_regime and bull_cross:
                    signals[i] = 0.25
                    position = 1
                elif is_bear_regime and bear_cross:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: WR reaches overbought (-20) or regime change to bear
            if wr >= overbought or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: WR reaches oversold (-80) or regime change to bull
            if wr <= oversold or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals