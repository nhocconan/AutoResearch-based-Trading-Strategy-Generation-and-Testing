#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R reversal + 1w EMA50 trend filter + volume spike
# Williams %R identifies overbought/oversold conditions; weekly EMA50 ensures trades align with higher timeframe trend;
# volume confirmation filters for strong moves. Designed to work in both bull (buy oversold in uptrend) and bear (sell overbought in downtrend).
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag while capturing mean-reversion within trends.

name = "1d_WilliamsR_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14) on daily data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: volume > 1.8x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, volume, Williams %R
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Williams %R rises above -20 (overbought)
            # 2. Price crosses below 1w EMA50 (trend change)
            # 3. Loss of volume confirmation (weakening momentum)
            if (curr_williams_r > -20 or
                curr_close < curr_ema_50_1w or
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Williams %R falls below -80 (oversold)
            # 2. Price crosses above 1w EMA50 (trend change)
            # 3. Loss of volume confirmation (weakening momentum)
            if (curr_williams_r < -80 or
                curr_close > curr_ema_50_1w or
                not curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Only enter with volume confirmation to avoid low-quality signals
            if not curr_volume_confirm:
                signals[i] = 0.0
                continue
                
            # Long entry: Williams %R below -80 (oversold) + price above 1w EMA50 (uptrend)
            if (curr_williams_r < -80 and
                curr_close > curr_ema_50_1w):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Williams %R above -20 (overbought) + price below 1w EMA50 (downtrend)
            elif (curr_williams_r > -20 and
                  curr_close < curr_ema_50_1w):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals