#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation.
# Enter long when Williams %R < -80 (oversold) and price > 1d EMA34 (uptrend) with volume spike (>2x 20-period average).
# Enter short when Williams %R > -20 (overbought) and price < 1d EMA34 (downtrend) with volume spike.
# Exit when Williams %R returns to -50 level or trend breaks.
# Designed for low trade frequency (~20-40/year) to minimize fee decay. Works in both bull and bear markets
# by combining mean-reversion entries with trend filtering to avoid counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Williams %R calculation (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 14-period Williams %R on 4h data
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], -50).fillna(-50).values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h Williams %R and 1d EMA to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        wr_val = williams_r_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict to reduce trades)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) + uptrend + volume spike
            if wr_val < -80 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) + downtrend + volume spike
            elif wr_val > -20 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R returns to -50 or trend breaks
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R > -50 or price breaks below EMA
                if wr_val > -50 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R < -50 or price breaks above EMA
                if wr_val < -50 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0