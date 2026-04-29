#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# Long when Williams %R(14) < -80 (oversold) AND close > 1d EMA50 AND volume > 1.5x average
# Short when Williams %R(14) > -20 (overbought) AND close < 1d EMA50 AND volume > 1.5x average
# Uses discrete sizing (0.25) and mean reversion logic to work in both bull and bear markets.
# Williams %R identifies exhaustion points; 1d EMA50 filters trend direction; volume confirms conviction.
# Timeframe: 4h (primary), HTF: 1d for EMA50 trend filter.

name = "4h_WilliamsR_MeanReversion_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 4h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and EMA50
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Williams %R rises above -50 (exiting oversold territory)
            # 2. Price crosses below 1d EMA50 (trend change)
            if (curr_williams_r > -50 or
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Williams %R falls below -50 (exiting overbought territory)
            # 2. Price crosses above 1d EMA50 (trend change)
            if (curr_williams_r < -50 or
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) AND close > 1d EMA50 AND volume confirm
            if (curr_williams_r < -80 and
                curr_close > curr_ema_50_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) AND close < 1d EMA50 AND volume confirm
            elif (curr_williams_r > -20 and
                  curr_close < curr_ema_50_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals