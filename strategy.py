#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) + 1d EMA34 Trend Filter + Volume Confirmation
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# Long when: Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d EMA34 AND volume > 1.5x avg
# Short when: Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d EMA34 AND volume > 1.5x avg
# Uses discrete sizing (0.25) to minimize fee churn. Works in bull/bear via 1d trend filter.
# Timeframe: 6h (primary), HTF: 1d for EMA34 trend.

name = "6h_ElderRay_1dEMA34_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop for 1d EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (using close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13, 20)  # warmup for EMA34, EMA13, volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Bull Power turns negative (momentum lost)
            # 2. Price falls below 1d EMA34 (trend change)
            if (curr_bull_power <= 0 or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Bear Power turns positive (momentum lost)
            # 2. Price rises above 1d EMA34 (trend change)
            if (curr_bear_power <= 0 or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power positive AND Bear Power rising (less negative) AND price > 1d EMA34 AND volume confirm
            if (curr_bull_power > 0 and
                i > start_idx and bear_power[i] > bear_power[i-1] and  # Bear Power rising
                curr_close > curr_ema_34_1d and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power negative AND Bull Power falling (less positive) AND price < 1d EMA34 AND volume confirm
            elif (curr_bear_power < 0 and
                  i > start_idx and bull_power[i] < bull_power[i-1] and  # Bull Power falling
                  curr_close < curr_ema_34_1d and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals