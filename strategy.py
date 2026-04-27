#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume spike.
# Williams %R measures overbought/oversold levels: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when %R crosses above -80 (oversold) in 1d uptrend with volume spike.
# Short when %R crosses below -20 (overbought) in 1d downtrend with volume spike.
# Uses 14-period lookback. Williams %R is effective in ranging markets and captures reversals.
# Combined with 1d trend filter to avoid counter-trend trades and volume spike for confirmation.
# Target: ~15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: Williams %R crosses above -80 (from below), 1d uptrend, volume filter
        if (williams_r[i] > -80 and williams_r[i-1] <= -80 and
            close[i] > ema34_1d_aligned[i] and
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: Williams %R crosses below -20 (from above), 1d downtrend, volume filter
        elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and
              close[i] < ema34_1d_aligned[i] and
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_1dEMA34_VolumeFilter"
timeframe = "6h"
leverage = 1.0