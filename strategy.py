#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Band Squeeze with 1-day Trend Filter and Volume Confirmation
# Long when BB Width < 20th percentile (squeeze) + price breaks above upper band + daily EMA(50) uptrend + volume spike
# Short when BB Width < 20th percentile (squeeze) + price breaks below lower band + daily EMA(50) downtrend + volume spike
# Bollinger Squeeze identifies low volatility periods preceding explosive moves
# Daily trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation in the breakout
# Targets 20-50 total trades over 4 years (5-12/year) to avoid fee drag

name = "4h_BollingerSqueeze_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std_dev)
    lower_band = sma - (bb_std * std_dev)
    bb_width = upper_band - lower_band
    
    # BB Width percentile (20-day lookback for squeeze detection)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).quantile(0.2).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(sma[i]) or 
            np.isnan(std_dev[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1d_val = ema50_1d_aligned[i]
        squeeze = squeeze_condition[i]
        vol_spike = volume_spike[i]
        upper = upper_band[i]
        lower = lower_band[i]
        close_val = close[i]
        
        if position == 0:
            # Enter long: squeeze + break above upper band + daily uptrend + volume spike
            if squeeze and close_val > upper and close_val > ema50_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze + break below lower band + daily downtrend + volume spike
            elif squeeze and close_val < lower and close_val < ema50_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below middle band OR daily trend turns down
            if close_val < sma[i] or close_val < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above middle band OR daily trend turns up
            if close_val > sma[i] or close_val > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals