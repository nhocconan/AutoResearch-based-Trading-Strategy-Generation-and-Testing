#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter
# - Long when BB width < 20th percentile (squeeze) AND price breaks above upper BB AND 1d close > 1d EMA50 (uptrend)
# - Short when BB width < 20th percentile (squeeze) AND price breaks below lower BB AND 1d close < 1d EMA50 (downtrend)
# - Exit when price returns to middle BB (20-period SMA)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Bollinger squeeze identifies low volatility periods prone to breakouts
# - Direction filter from 1d EMA ensures we trade with higher timeframe trend
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_bb_squeeze_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute 6h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Pre-compute 6h BB width percentile (20-period lookback)
    def rolling_percentile(arr, window, percentile):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            window_data = arr[i - window + 1:i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                result[i] = np.percentile(valid_data, percentile)
            else:
                result[i] = np.nan
        return result
    
    bb_width_percentile = rolling_percentile(bb_width, 50, 20)  # 50-period lookback for 20th percentile
    squeeze_condition = bb_width < bb_width_percentile  # BB width below 20th percentile
    
    # Pre-compute 6h middle BB (20-period SMA) for exit
    middle_bb = sma_20
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(squeeze_condition[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(middle_bb[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: BB squeeze AND price breaks above upper BB AND 1d uptrend
            if squeeze_condition[i] and close[i] > upper_bb[i] and close > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short conditions: BB squeeze AND price breaks below lower BB AND 1d downtrend
            elif squeeze_condition[i] and close[i] < lower_bb[i] and close < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to middle BB
            exit_long = (position == 1 and close[i] <= middle_bb[i])
            exit_short = (position == -1 and close[i] >= middle_bb[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals