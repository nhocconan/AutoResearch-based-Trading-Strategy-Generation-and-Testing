#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation
    # Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
    # Long when Bull Power > 0 and rising + price > 1w EMA50 + volume spike
    # Short when Bear Power > 0 and rising + price < 1w EMA50 + volume spike
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year)
    # Elder Ray shows strength of bulls/bears relative to trend, effective in both bull/bear markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Elder Ray calculations (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power = ema_13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # Align 1d Elder Ray to 1d (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1w (wait for completed 1w bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray momentum: rising Bull/Bear Power (use previous bar to avoid look-ahead)
        if i >= 101:
            bull_prev = bull_power_aligned[i-1]
            bull_curr = bull_power_aligned[i]
            bear_prev = bear_power_aligned[i-1]
            bear_curr = bear_power_aligned[i]
            bull_rising = bull_curr > bull_prev
            bear_rising = bear_curr > bear_prev
        else:
            bull_rising = False
            bear_rising = False
        
        # Volume confirmation: current 1d volume > 2.0 * 20-period average
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_confirm = vol_1d_aligned[i] > 2.0 * vol_ma_aligned[i]
        
        # Trend filter: price vs 1w EMA50
        price_above_1w_ema = close[i] > ema_50_aligned[i]
        price_below_1w_ema = close[i] < ema_50_aligned[i]
        
        # Long signals: Bull Power rising + price above 1w EMA + volume confirmation
        long_entry = bull_rising and price_above_1w_ema and volume_confirm
        
        # Short signals: Bear Power rising + price below 1w EMA + volume confirmation
        short_entry = bear_rising and price_below_1w_ema and volume_confirm
        
        # Exit conditions: Elder Ray reversal or trend change
        long_exit = (not bull_rising) or (not price_above_1w_ema) or (position == 1 and bear_power_aligned[i] > bull_power_aligned[i])
        short_exit = (not bear_rising) or (not price_below_1w_ema) or (position == -1 and bull_power_aligned[i] > bear_power_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0