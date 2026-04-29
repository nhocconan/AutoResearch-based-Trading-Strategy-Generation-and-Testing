#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion + 1d EMA34 trend filter + volume spike confirmation
# Williams %R identifies overbought/oversold conditions; 1d EMA34 provides higher timeframe trend direction;
# volume spike confirms momentum behind the mean reversion move. Works in both bull and bear markets
# by fading extremes in the direction of the higher timeframe trend. Target: 20-40 trades/year.

name = "6h_WilliamsR_1dEMA34_VolumeSpike_MeanRev_v1"
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
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 2.0x 24-period average (4h equivalent on 6h chart)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24, 14)  # warmup for EMA34, volume, Williams %R
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit when Williams %R rises above -20 (overbought) or trend changes
            if curr_williams_r > -20 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -80 (oversold) or trend changes
            if curr_williams_r < -80 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter with volume confirmation to avoid weak moves
            if not curr_volume_confirm:
                signals[i] = 0.0
                continue
                
            # Long entry: Williams %R below -80 (oversold) and price above 1d EMA34 (uptrend)
            if curr_williams_r < -80 and curr_close > curr_ema_34_1d:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R above -20 (overbought) and price below 1d EMA34 (downtrend)
            elif curr_williams_r > -20 and curr_close < curr_ema_34_1d:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals