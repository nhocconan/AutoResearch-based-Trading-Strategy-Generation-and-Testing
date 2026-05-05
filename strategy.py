#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) AND BB width < 20th percentile (squeeze) AND price > EMA50(12h) AND volume > 1.5x 20-period average
# Short when price breaks below lower BB(20,2) AND BB width < 20th percentile (squeeze) AND price < EMA50(12h) AND volume > 1.5x 20-period average
# Exit when price returns to middle BB(20) OR trend flips (price crosses EMA50(12h))
# Bollinger squeeze identifies low volatility periods primed for explosive moves
# 12h EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws
# Volume spike confirms institutional participation in the breakout
# Target: 12-37 trades/year per symbol (50-150 total over 4 years)
# Discrete sizing (0.25) to limit fee drag

name = "6h_BollingerSqueeze_Breakout_12hEMA50_Trend_VolumeSpike"
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
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Bollinger Bands (20,2) on 6h close
    if len(close) >= 20:
        sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = sma_20 + (2.0 * std_20)
        lower_bb = sma_20 - (2.0 * std_20)
        middle_bb = sma_20
        bb_width = (upper_bb - lower_bb) / middle_bb  # Normalized width
    else:
        sma_20 = np.full(n, np.nan)
        std_20 = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        middle_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window)
    bb_width_percentile_20 = np.full(n, np.nan)
    for i in range(20, n):
        bb_width_percentile_20[i] = np.percentile(bb_width[20:i+1], 20)
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(sma_20[i]) or 
            np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or 
            np.isnan(bb_width[i]) or 
            np.isnan(bb_width_percentile_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper BB AND BB width < 20th percentile (squeeze) AND price > EMA50(12h) AND volume spike
            if (close[i] > upper_bb[i] and 
                bb_width[i] < bb_width_percentile_20[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower BB AND BB width < 20th percentile (squeeze) AND price < EMA50(12h) AND volume spike
            elif (close[i] < lower_bb[i] and 
                  bb_width[i] < bb_width_percentile_20[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle BB OR price < EMA50(12h) (trend flip)
            if (close[i] < middle_bb[i] or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle BB OR price > EMA50(12h) (trend flip)
            if (close[i] > middle_bb[i] or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals