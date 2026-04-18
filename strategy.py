#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Mean Reversion + 12h Trend Filter + Volume Spike
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# 12h EMA50 trend filter ensures trades align with higher timeframe trend.
# Volume spike confirms institutional participation at reversal points.
# Works in ranging markets (mean reversion) and trending markets (pullbacks to EMA).
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
name = "6h_WilliamsR_MeanReversion_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Williams %R on 6h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0 * 24-period average (4 days on 6h chart)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        ema_trend = ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Oversold (Williams %R < -80) + above 12h EMA50 + volume spike
            if wr < -80 and close[i] > ema_trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (Williams %R > -20) + below 12h EMA50 + volume spike
            elif wr > -20 and close[i] < ema_trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion complete) or below 12h EMA50 (trend break)
            if wr > -50 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion complete) or above 12h EMA50 (trend break)
            if wr < -50 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals