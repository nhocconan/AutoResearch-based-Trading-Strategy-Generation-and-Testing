#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d CCI(20) mean reversion with weekly trend filter
# Long when CCI crosses above -100 (oversold recovery) and weekly EMA50 is rising
# Short when CCI crosses below +100 (overbought rejection) and weekly EMA50 is falling
# Exit when CCI crosses zero (mean reversion complete) or trend changes
# Targets 20-40 trades per year to minimize fee drag while capturing mean reversion in ranging markets
# CCI identifies overextended moves, weekly EMA50 filters counter-trend noise in both bull/bear markets

name = "1d_CCI20_WeeklyEMA50_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_std = pd.Series(typical_price).rolling(window=20, min_periods=20).std().values
    # Avoid division by zero
    tp_std = np.where(tp_std == 0, 1e-10, tp_std)
    cci = (typical_price - tp_ma) / (0.015 * tp_std)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly close for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_slope = ema50_weekly[1:] - ema50_weekly[:-1]  # slope: positive = uptrend
    ema50_weekly_slope = np.concatenate([[0], ema50_weekly_slope])  # align length
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    ema50_weekly_slope_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for CCI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(cci[i]) or np.isnan(ema50_weekly_aligned[i]) or 
            np.isnan(ema50_weekly_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        cci_val = cci[i]
        ema50_slope = ema50_weekly_slope_aligned[i]
        
        if position == 0:
            # Enter long: CCI crosses above -100 from below (oversold recovery) with weekly uptrend
            if cci_val > -100 and cci[i-1] <= -100 and ema50_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: CCI crosses below +100 from above (overbought rejection) with weekly downtrend
            elif cci_val < 100 and cci[i-1] >= 100 and ema50_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: CCI crosses above zero (mean reversion) or weekly trend turns down
            if cci_val > 0 and cci[i-1] <= 0 or ema50_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: CCI crosses below zero (mean reversion) or weekly trend turns up
            if cci_val < 0 and cci[i-1] >= 0 or ema50_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals