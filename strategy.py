#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h price action with 4h trend filter and 1d volume spike filter
# - Uses 4h EMA20 for trend direction (long above EMA20, short below EMA20)
# - Uses 1h price action: breakout above recent 1h high for long, below recent 1h low for short
# - Uses 1d volume spike (>1.5x 20-day average) to confirm momentum
# - Only trades during active session (08-20 UTC) to avoid low-liquidity noise
# - Fixed position size of 0.20 to control risk and minimize fee churn
# - Target: 15-30 trades/year to stay within fee limits

name = "1h_Trend_Breakout_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * vol_ma20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1h price action: recent high/low for breakout
    # Use 5-period lookback for recent swing high/low
    high_roll5 = pd.Series(high).rolling(window=5, min_periods=5).max().values
    low_roll5 = pd.Series(low).rolling(window=5, min_periods=5).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough history for indicators
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(high_roll5[i]) or 
            np.isnan(low_roll5[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above recent 5-period high, 4h uptrend, volume spike
            long_cond = (close[i] > high_roll5[i-1] and 
                        ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1] and
                        vol_spike_1d_aligned[i])
            
            # Short: price breaks below recent 5-period low, 4h downtrend, volume spike
            short_cond = (close[i] < low_roll5[i-1] and 
                         ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1] and
                         vol_spike_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below recent 5-period low
            if close[i] < low_roll5[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above recent 5-period high
            if close[i] > high_roll5[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals