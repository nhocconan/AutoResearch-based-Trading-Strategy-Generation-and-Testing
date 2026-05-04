#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 from oversold with 1d EMA50 uptrend and volume > 1.5x 20-period volume EMA
# Short when Williams %R(14) crosses below -20 from overbought with 1d EMA50 downtrend and volume > 1.5x 20-period volume EMA
# Uses 1d HTF for trend to reduce whipsaw vs shorter HTF, targeting 15-35 trades/year on 6h.
# Volume spike filter (1.5x) is strict to avoid overtrading. Williams %R provides mean reversion edge in ranging markets.
# Works in bull markets via longs on pullbacks and bear markets via shorts on rallies.

name = "6h_WilliamsR_1dEMA50_Trend_VolumeSpike"
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
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) - ONCE before loop
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_williams_r = williams_r[0] if n > 0 else 0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(prev_williams_r)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            prev_williams_r = williams_r[i] if not np.isnan(williams_r[i]) else prev_williams_r
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (oversold) AND 1d uptrend AND volume spike
            if (williams_r[i] > -80 and prev_williams_r <= -80 and 
                close[i] > ema_50_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 (overbought) AND 1d downtrend AND volume spike
            elif (williams_r[i] < -20 and prev_williams_r >= -20 and 
                  close[i] < ema_50_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR 1d trend turns down
            if (williams_r[i] > -20 and prev_williams_r <= -20) or \
               (close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR 1d trend turns up
            if (williams_r[i] < -80 and prev_williams_r >= -80) or \
               (close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        prev_williams_r = williams_r[i]
    
    return signals