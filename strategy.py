#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA34 trend filter and volume spike confirmation.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d EMA34 AND volume > 2x 20-bar average.
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d EMA34 AND volume > 2x 20-bar average.
# Elder Ray measures bull/bear strength relative to trend (EMA13), 1d EMA34 filters for higher timeframe trend alignment, volume spike confirms conviction.
# Works in bull (buy strength in uptrend) and bear (sell weakness in downtrend) by trading with aligned 1d trend.
# Target: 12-37 trades/year on 6h (50-150 total over 4 years). Discrete sizing 0.25 to minimize fee drag.

name = "6h_ElderRay_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h EMA13 for Elder Ray
    close_6h = df_6h['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_6h['high'].values - ema_13_6h
    bear_power = df_6h['low'].values - ema_13_6h
    
    # Align Elder Ray components to 6h primary timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Calculate 6h EMA13 for trend alignment (used in exit)
    ema_13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_13_6h)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h primary timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for EMA13 (13) + EMA34 (34)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_13_6h_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_bull_power = bull_power_aligned[i]
        curr_bear_power = bear_power_aligned[i]
        curr_ema_13_6h = ema_13_6h_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Volume confirmation: current 6h volume > 2x 20-period average
        vol_6h = df_6h['volume'].values
        vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
        vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
        curr_vol_ma = vol_ma_6h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (bulls in control) AND Bear Power rising (less negative) AND price > 1d EMA34 AND volume confirmation
            if (curr_bull_power > 0 and 
                curr_bear_power > bear_power_aligned[i-1] and  # Bear Power rising (less negative)
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) AND Bull Power falling (less positive) AND price < 1d EMA34 AND volume confirmation
            elif (curr_bear_power < 0 and 
                  curr_bull_power < bull_power_aligned[i-1] and  # Bull Power falling (less positive)
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 (bulls lose control) OR price < 6h EMA13 (trend violation)
            if (curr_bull_power <= 0 or 
                curr_close < curr_ema_13_6h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (bears lose control) OR price > 6h EMA13 (trend violation)
            if (curr_bear_power >= 0 or 
                curr_close > curr_ema_13_6h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals