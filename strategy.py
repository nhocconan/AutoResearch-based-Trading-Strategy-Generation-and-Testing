#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w EMA50 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND close > 1w EMA50 (uptrend) AND volume > 1.5 * 20-bar avg volume
# Short when Williams %R > -20 (overbought) AND close < 1w EMA50 (downtrend) AND volume > 1.5 * 20-bar avg volume
# Exit when Williams %R crosses above -50 for longs or below -50 for shorts (mean reversion)
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R identifies extreme price levels; 1w EMA50 ensures alignment with weekly trend
# Volume spike confirms institutional participation; mean-reversion exit works in ranging markets

name = "6h_WilliamsR_1wEMA50_Volume_v1"
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
    
    # Calculate Williams %R (14-period) using 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Williams %R signals with trend and volume filters
            # Long: Williams %R < -80 (oversold) AND uptrend AND volume spike
            if williams_r[i] < -80 and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND downtrend AND volume spike
            elif williams_r[i] > -20 and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (mean reversion)
            if williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (mean reversion)
            if williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals