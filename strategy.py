#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze with 1d Trend Filter and Volume Confirmation
# Bollinger Bands squeeze (low volatility) indicates impending breakout.
# Daily trend filter ensures we only take breakouts in direction of higher timeframe trend.
# Volume confirmation validates breakout strength.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.
# Works in both bull and bear markets by filtering for trend direction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h Bollinger Bands (20, 2.0)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Squeeze: width below 20-period average width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Daily trend: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_uptrend = ema_50_1d > ema_200_1d
    daily_downtrend = ema_50_1d < ema_200_1d
    
    # Daily volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 6h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition.astype(float))
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(squeeze_aligned[i]) or np.isnan(daily_uptrend_aligned[i]) or
            np.isnan(daily_downtrend_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x daily volume MA (adjusted for 6h)
        # 4 6h periods per day, so daily MA/4 = approximate 6h period MA
        volume_6h_approx_ma = volume_ma_20_1d_aligned[i] / 4
        volume_condition = volume[i] > (volume_6h_approx_ma * 1.5)
        
        # Bollinger Band breakout conditions
        breakout_up = close[i] > bb_upper[i]
        breakout_down = close[i] < bb_lower[i]
        
        # Entry conditions: Bollinger Band breakout with squeeze, trend alignment, and volume
        if position == 0:
            if squeeze_aligned[i] > 0.5 and breakout_up and daily_uptrend_aligned[i] > 0.5 and volume_condition:
                position = 1
                signals[i] = position_size
            elif squeeze_aligned[i] > 0.5 and breakout_down and daily_downtrend_aligned[i] > 0.5 and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price returns to middle band or trend changes
            if close[i] < bb_middle[i] or daily_downtrend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price returns to middle band or trend changes
            if close[i] > bb_middle[i] or daily_uptrend_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Bollinger_Squeeze_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0