#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume spike
# Williams %R identifies overbought/oversold conditions (80/20 levels) for mean reversion entries
# 6h timeframe balances trade frequency and signal quality for 6h strategies
# 1d EMA(34) provides trend filter to only take reversals in direction of higher timeframe trend
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) minimizes fee drag while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Williams %R works in both bull and bear markets by fading extremes only when aligned with 1d trend
# Avoids overtrading by requiring confluence of 3 conditions: %R extreme + trend + volume

name = "6h_WilliamsR_Reversal_1dEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(34) for trend determination
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 6h data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We use 14-period lookback
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero with neutral value
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) + volume spike + close > 1d EMA34 (bullish trend)
            if williams_r[i] < -80 and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + volume spike + close < 1d EMA34 (bearish trend)
            elif williams_r[i] > -20 and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) or close < 1d EMA34 (trend reversal)
            if williams_r[i] > -20 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) or close > 1d EMA34 (trend reversal)
            if williams_r[i] < -80 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals