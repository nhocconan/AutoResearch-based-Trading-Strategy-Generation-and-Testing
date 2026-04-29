#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion + 12h EMA50 trend filter + volume spike confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# 12h EMA50 provides higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation ensures breakout/breakdown has institutional participation.
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag while capturing reversals.

name = "6h_WilliamsR_12hEMA50_VolumeSpike_MeanRev_v1"
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
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14-period) on 6h timeframe
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Calculate volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, volume, Williams %R
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Williams %R returns above -20 (overbought)
            # 2. Price crosses below 12h EMA50 (trend change)
            # 3. Volume confirmation lost (institutional interest fading)
            if (curr_williams_r > -20 or
                curr_close < curr_ema_50_12h or
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Williams %R returns below -80 (oversold)
            # 2. Price crosses above 12h EMA50 (trend change)
            # 3. Volume confirmation lost (institutional interest fading)
            if (curr_williams_r < -80 or
                curr_close > curr_ema_50_12h or
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter with volume confirmation to avoid false signals
            if not curr_volume_confirm:
                signals[i] = 0.0
                continue
                
            # Long entry: Williams %R below -80 (oversold) + above 12h EMA50 (uptrend filter)
            if (curr_williams_r < -80 and
                curr_close > curr_ema_50_12h):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Williams %R above -20 (overbought) + below 12h EMA50 (downtrend filter)
            elif (curr_williams_r > -20 and
                  curr_close < curr_ema_50_12h):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals