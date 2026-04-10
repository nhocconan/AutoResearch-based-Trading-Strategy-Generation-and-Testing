#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume spike confirmation
# - Long when 6h Williams %R < -80 (oversold) AND 12h EMA50 > EMA200 (bullish trend) AND 6h volume > 1.5x 20-period volume SMA
# - Short when 6h Williams %R > -20 (overbought) AND 12h EMA50 < EMA200 (bearish trend) AND 6h volume > 1.5x 20-period volume SMA
# - Exit: Williams %R returns to -50 level or volume drops below average
# - Position sizing: 0.25 discrete level to minimize fee drag
# - Target: 12-37 trades/year on 6h timeframe to stay within fee drag limits
# - Williams %R identifies extreme reversals, 12h EMA crossover filters trend direction, volume confirms momentum

name = "6h_12h_williamsr_meanreversion_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 6h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate 12h EMA50 and EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate 12h close for trend comparison
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Calculate 6h volume SMA for confirmation
    volume_sma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(ema_200_12h_aligned[i]) or np.isnan(close_12h_aligned[i]) or
            np.isnan(volume_sma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20_6h[i]
        
        # Trend filter: 12h EMA50 vs EMA200
        trend_bullish = ema_50_12h_aligned[i] > ema_200_12h_aligned[i]
        trend_bearish = ema_50_12h_aligned[i] < ema_200_12h_aligned[i]
        
        # Williams %R mean reversion signals
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_level = abs(williams_r[i] + 50) < 10  # Near -50 level
        
        if position == 0:  # Flat - look for entry
            if oversold and trend_bullish and vol_confirm:
                position = 1
                signals[i] = 0.25
            elif overbought and trend_bearish and vol_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_level or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_level or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals