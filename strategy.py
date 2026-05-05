#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversion with 1w EMA50 trend filter and volume confirmation
# Williams %R(14) measures overbought/oversold on 6h timeframe
# Long when Williams %R < -80 (extreme oversold) AND price > 1w EMA50 (uptrend filter) AND volume > 1.5x 24-period average
# Short when Williams %R > -20 (extreme overbought) AND price < 1w EMA50 (downtrend filter) AND volume > 1.5x 24-period average
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts) OR trend flips (price crosses 1w EMA50)
# Williams %R is effective in ranging markets and captures reversals after extreme moves
# 1w EMA50 provides higher timeframe trend filter to avoid counter-trend whipsaws
# Volume spike confirms institutional participation during reversion
# Target: 12-37 trades/year per symbol (50-150 total over 4 years) for 6h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "6h_WilliamsR_EXTREME_1wEMA50_Trend_VolumeSpike"
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
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R(14) on 6h timeframe
    if len(high) >= 14:
        # Calculate highest high and lowest low over last 14 periods
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        # Avoid division by zero
        hl_range = highest_high - lowest_low
        williams_r = np.where(hl_range != 0, -100 * (highest_high - close) / hl_range, -50.0)
    else:
        williams_r = np.full(n, -50.0)
    
    # Volume confirmation: volume > 1.5x 24-period average (spike filter)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (1.5 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (extreme oversold) AND price > 1w EMA50 AND volume spike
            if (williams_r[i] < -80.0 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (extreme overbought) AND price < 1w EMA50 AND volume spike
            elif (williams_r[i] > -20.0 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (reversion complete) OR price < 1w EMA50 (trend flip)
            if (williams_r[i] > -50.0 or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (reversion complete) OR price > 1w EMA50 (trend flip)
            if (williams_r[i] < -50.0 or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals