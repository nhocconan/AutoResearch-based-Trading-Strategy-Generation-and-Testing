#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R(14) with 1w EMA34 trend filter and volume confirmation
# Long when Williams %R crosses above -80 (oversold) with 1w EMA34 rising and volume > 1.5x 24-period avg
# Short when Williams %R crosses below -20 (overbought) with 1w EMA34 falling and volume > 1.5x 24-period avg
# Exit when Williams %R crosses opposite threshold (-20 for long, -80 for short) or trend reverses
# Williams %R identifies mean reversion extremes, EMA34 filters trend direction, volume confirms momentum
# Targets 12-37 trades per year for optimal fee drag (< 200 total over 4 years)

name = "12h_WilliamsR_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 12h data
    def williams_r(high_arr, low_arr, close_arr, window):
        wr = np.full_like(close_arr, np.nan, dtype=float)
        for i in range(window - 1, len(close_arr)):
            highest_high = np.max(high_arr[i - window + 1:i + 1])
            lowest_low = np.min(low_arr[i - window + 1:i + 1])
            if highest_high != lowest_low:
                wr[i] = -100 * (highest_high - close_arr[i]) / (highest_high - lowest_low)
            else:
                wr[i] = -50  # avoid division by zero
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_slope = ema34_1w[1:] - ema34_1w[:-1]  # slope: positive = uptrend
    ema34_1w_slope = np.concatenate([[0], ema34_1w_slope])  # align length
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    ema34_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w_slope)
    
    # Volume confirmation: current volume > 1.5x 24-period average (2 * 12h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(ema34_1w_slope_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr_val = wr[i]
        ema34_val = ema34_1w_aligned[i]
        ema34_slope = ema34_1w_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 (oversold), volume confirmation, 1w uptrend
            if i > start_idx and wr[i-1] <= -80 and wr_val > -80 and vol_conf_val and ema34_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 (overbought), volume confirmation, 1w downtrend
            elif i > start_idx and wr[i-1] >= -20 and wr_val < -20 and vol_conf_val and ema34_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) or 1w trend turns down
            if wr_val > -20 or ema34_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) or 1w trend turns up
            if wr_val < -80 or ema34_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals