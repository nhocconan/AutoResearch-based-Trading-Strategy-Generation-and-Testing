#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1w EMA trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion
# 1w EMA filter ensures we only trade in direction of weekly trend (avoid counter-trend)
# Volume confirmation ensures breakouts/retracements have participation
# Works in bull/bear: trend filter adapts, Williams %R captures pullbacks in trends
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_1w_williamsr_volume_v1"
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
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close) / (highest_high - lowest_low),
                          -50)  # neutral when range is zero
    
    # Calculate 6h average volume (20-period) for confirmation
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x 6h average volume
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        # Trend filter: price above/below weekly EMA
        above_weekly_ema = close[i] > ema_34_1w_aligned[i]
        below_weekly_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR volume fails
            if williams_r[i] > -20 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR volume fails
            if williams_r[i] < -80 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: mean reversion in direction of weekly trend
            if above_weekly_ema and volume_confirmed:
                # In uptrend: look for oversold pullbacks to go long
                if williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
            elif below_weekly_ema and volume_confirmed:
                # In downtrend: look for overbought bounces to go short
                if williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals