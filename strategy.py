#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels (>80 or <20) with volume
# indicate high-probability mean reversion spots. 1d EMA50 provides trend filter to trade in direction of higher timeframe momentum
# and reduce counter-trend whipsaws. Volume spike (1.8x 24-period average) confirms reversal conviction.
# Targets 80-180 trades over 4 years (20-45/year) for 6h timeframe, balancing opportunity with cost control.
# Works in bull markets by buying oversold dips in uptrends and in bear markets by selling overbought rallies in downtrends.

name = "6h_WilliamsR_Reversal_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for EMA and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate rolling highest high and lowest low for 1d
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when HH == LL
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate volume spike (1.8x 24-period average on 6h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R, EMA, and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 (oversold reversal) + price > 1d EMA50 + volume spike
            if williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal) + price < 1d EMA50 + volume spike
            elif williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) or crosses below -80 (strong oversold continuation)
            if williams_r_aligned[i] > -20 or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) or crosses above -20 (strong overbought continuation)
            if williams_r_aligned[i] < -80 or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals