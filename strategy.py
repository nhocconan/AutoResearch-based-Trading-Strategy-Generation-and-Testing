#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d EMA trend filter and volume confirmation
# Long when BB width < 20th percentile (squeeze) AND price breaks above upper band AND 1d EMA50 uptrend AND volume spike
# Short when BB width < 20th percentile (squeeze) AND price breaks below lower band AND 1d EMA50 downtrend AND volume spike
# Bollinger squeeze identifies low volatility periods preceding breakouts
# Works in bull markets (continuation breakouts) and bear markets (mean-reversion bounces or breakdowns)
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_BB_Squeeze_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (4h timeframe)
    bb_period = 20
    bb_std = 2.0
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = bb_ma + (bb_std * bb_std_dev)
    lower_band = bb_ma - (bb_std * bb_std_dev)
    bb_width = (upper_band - lower_band) / bb_ma  # Normalized width
    
    # Calculate percentile rank of BB width (20-period lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 50)  # BB period and percentile lookback
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_width_pct = bb_width_percentile[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # BB squeeze condition: width in lowest 20% of recent values
        is_squeeze = curr_width_pct <= 0.20
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below middle band OR BB squeeze ends (volatility expansion)
            if curr_close < bb_ma[i] or not is_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above middle band OR BB squeeze ends
            if curr_close > bb_ma[i] or not is_squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: BB squeeze AND price breaks above upper band AND 1d EMA50 uptrend AND volume spike
            if (is_squeeze and 
                curr_close > curr_upper and 
                curr_close > curr_ema_1d and  # price above 1d EMA50 for uptrend
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: BB squeeze AND price breaks below lower band AND 1d EMA50 downtrend AND volume spike
            elif (is_squeeze and 
                  curr_close < curr_lower and 
                  curr_close < curr_ema_1d and  # price below 1d EMA50 for downtrend
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals