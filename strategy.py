#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation
    # Long when Bull Power > 0 (close > EMA13) + 12h EMA50 uptrend + volume > 1.5x average
    # Short when Bear Power < 0 (close < EMA13) + 12h EMA50 downtrend + volume > 1.5x average
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year)
    # Elder Ray measures bull/bear power relative to EMA; trend filter avoids counter-trend trades
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13
    bull_power = close - ema_13
    # Bear Power = EMA13 - Close
    bear_power = ema_13 - close
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h volume average (20-period) with min_periods
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    vol_ma_20_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Current 12h volume for confirmation
    vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or
            np.isnan(vol_12h_current[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        bull_power_positive = bull_power_val > 0
        bear_power_negative = bear_power_val > 0  # Bear Power positive means bearish
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period average
        volume_confirm = vol_12h_current[i] > 1.5 * vol_ma_12h_aligned[i]
        
        # Trend filter: 12h EMA50 slope (using previous bar to avoid look-ahead)
        if i >= 1:
            ema_prev = ema_50_12h_aligned[i-1]
            ema_curr = ema_50_12h_aligned[i]
            ema_slope_up = ema_curr > ema_prev
            ema_slope_down = ema_curr < ema_prev
        else:
            ema_slope_up = False
            ema_slope_down = False
        
        # Entry signals
        long_entry = bull_power_positive and ema_slope_up and volume_confirm
        short_entry = bear_power_negative and ema_slope_down and volume_confirm
        
        # Exit conditions: power reversal or trend change
        long_exit = bull_power_val <= 0 or not ema_slope_up
        short_exit = bear_power_val <= 0 or not ema_slope_down
        
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

name = "6h_12h_elder_ray_trend_volume_v1"
timeframe = "6h"
leverage = 1.0