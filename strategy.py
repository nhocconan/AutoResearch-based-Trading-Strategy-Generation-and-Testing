#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R (14) extreme + 1w EMA(34) trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion in ranging markets
# 1w EMA(34) provides major trend filter: only take long signals above EMA, short below
# Volume confirmation ensures breakout/reversal has participation
# Works in bull/bear: trend filter adapts to major direction, Williams %R captures reversals
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25

name = "1d_1w_williamsr_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate 1d average volume (20-period) for confirmation
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR trend turns bearish
            if williams_r[i] > -20 or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR trend turns bullish
            if williams_r[i] < -80 or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extremes with volume confirmation and trend filter
            if williams_r[i] < -80 and volume_confirmed and close[i] > ema_1w_aligned[i]:
                # Oversold + volume + bullish trend = long
                position = 1
                signals[i] = 0.25
            elif williams_r[i] > -20 and volume_confirmed and close[i] < ema_1w_aligned[i]:
                # Overbought + volume + bearish trend = short
                position = -1
                signals[i] = -0.25
    
    return signals