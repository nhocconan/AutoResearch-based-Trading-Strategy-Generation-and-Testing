#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes breakouts in both bull and bear markets
# Long when: BB width < 20th percentile AND close > upper band AND 1d close > 1d EMA50 AND volume > 2x 20-period MA
# Short when: BB width < 20th percentile AND close < lower band AND 1d close < 1d EMA50 AND volume > 2x 20-period MA
# Exit when: BB width > 50th percentile (squeeze ends) OR opposite band touch
# Uses BB squeeze for low volatility breakouts, 1d EMA for trend filter, volume for conviction
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_BBSqueeze_1dEMA50_VolumeBreakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 4h
    if len(close) >= 20:
        ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_band = ma_20 + (2 * std_20)
        lower_band = ma_20 - (2 * std_20)
        bb_width = (upper_band - lower_band) / ma_20  # normalized width
        
        # Percentile rank of BB width (20-period lookback)
        bb_width_percentile = np.full(n, np.nan)
        for i in range(20, n):
            window = bb_width[max(0, i-19):i+1]
            if not np.all(np.isnan(window)):
                bb_width_percentile[i] = (np.sum(~np.isnan(window) & (window <= bb_width[i])) / np.sum(~np.isnan(window))) * 100
        
        squeeze_condition = bb_width_percentile < 20  # BB width in lower 20th percentile
        long_breakout = close > upper_band
        short_breakout = close < lower_band
    else:
        ma_20 = np.full(n, np.nan)
        std_20 = np.full(n, np.nan)
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
        bb_width_percentile = np.full(n, np.nan)
        squeeze_condition = np.zeros(n, dtype=bool)
        long_breakout = np.zeros(n, dtype=bool)
        short_breakout = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
        trend_up = close_1d > ema_50_1d
        trend_down = close_1d < ema_50_1d
    else:
        ema_50_1d = np.full(len(close_1d), np.nan)
        trend_up = np.zeros(len(close_1d), dtype=bool)
        trend_down = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d trend to 4h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: squeeze + long breakout + uptrend + volume
            if (squeeze_condition[i] and 
                long_breakout[i] and 
                trend_up_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: squeeze + short breakout + downtrend + volume
            elif (squeeze_condition[i] and 
                  short_breakout[i] and 
                  trend_down_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: squeeze ends OR short breakout
            if (bb_width_percentile[i] > 50 or short_breakout[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: squeeze ends OR long breakout
            if (bb_width_percentile[i] > 50 or long_breakout[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals