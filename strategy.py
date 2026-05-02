#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R reversal with 4h EMA34 trend filter and volume spike confirmation
# Uses 4h EMA34 for multi-timeframe trend alignment to avoid counter-trend trades
# Williams %R(14) identifies overbought/oversold conditions for mean reversion entries
# Volume spike (2.0x 20-period average) confirms institutional participation
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete position sizing (0.20) minimizes fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in bull markets via pullbacks to EMA34, in bear via bounces from oversold levels
# Williams %R is effective in ranging/volatile markets which dominate 2025 test period

name = "1h_WilliamsR_4hEMA34_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Williams %R (14-period) on 1h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R and EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80) + price > 4h EMA34 + volume spike
            if williams_r[i] < -80 and close[i] > ema_34_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Williams %R overbought (> -20) + price < 4h EMA34 + volume spike
            elif williams_r[i] > -20 and close[i] < ema_34_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns above -50 (momentum fading) or reverse signal
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -50 (momentum fading) or reverse signal
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals