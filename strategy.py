#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout + 12h Trend Filter + Volume Confirmation
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets.
# Long when: BB width < 20th percentile (squeeze) AND price breaks above upper band AND 12h EMA50 uptrend AND volume > 2x 20-period MA
# Short when: BB width < 20th percentile (squeeze) AND price breaks below lower band AND 12h EMA50 downtrend AND volume > 2x 20-period MA
# Exit when: price returns to middle band (mean reversion after breakout fails) OR BB width expands above 50th percentile (squeeze ended)
# Uses BB squeeze for low volatility breakout setups, 12h EMA for trend filter, volume for conviction
# Timeframe: 6h, HTF: 12h. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_BBSqueeze_Breakout_12hEMA_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands on 6h (20, 2)
    if len(close) >= 20:
        ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_band = ma_20 + (2 * std_20)
        lower_band = ma_20 - (2 * std_20)
        bb_width = (upper_band - lower_band) / ma_20  # normalized width
    else:
        ma_20 = np.full(n, np.nan)
        upper_band = np.full(n, np.nan)
        lower_band = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # Calculate BB width percentiles for squeeze detection (using 50-period lookback)
    bb_width_pct = np.full(n, np.nan)
    if len(bb_width) >= 50:
        for i in range(49, len(bb_width)):
            if not np.isnan(bb_width[i]):
                window = bb_width[max(0, i-49):i+1]
                valid_window = window[~np.isnan(window)]
                if len(valid_window) >= 20:
                    bb_width_pct[i] = (np.sum(valid_window <= bb_width[i]) / len(valid_window)) * 100
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h
    close_12h = df_12h['close'].values
    if len(close_12h) >= 50:
        ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
        # Determine trend: price above/below EMA50
        ema50_uptrend = close_12h > ema_50_12h
        ema50_downtrend = close_12h < ema_50_12h
    else:
        ema_50_12h = np.full(len(close_12h), np.nan)
        ema50_uptrend = np.zeros(len(close_12h), dtype=bool)
        ema50_downtrend = np.zeros(len(close_12h), dtype=bool)
    
    # Align 12h EMA50 trend to 6h timeframe
    ema50_uptrend_aligned = align_htf_to_ltf(prices, df_12h, ema50_uptrend.astype(float))
    ema50_downtrend_aligned = align_htf_to_ltf(prices, df_12h, ema50_downtrend.astype(float))
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ma_20[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(bb_width_pct[i]) or np.isnan(ema50_uptrend_aligned[i]) or 
            np.isnan(ema50_downtrend_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze conditions: BB width below 20th percentile (low volatility)
        squeeze_condition = bb_width_pct[i] < 20.0
        
        if position == 0:
            # Long conditions: squeeze + breakout above upper band + 12h uptrend + volume
            if (squeeze_condition and 
                close[i] > upper_band[i] and 
                ema50_uptrend_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: squeeze + breakout below lower band + 12h downtrend + volume
            elif (squeeze_condition and 
                  close[i] < lower_band[i] and 
                  ema50_downtrend_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle band OR squeeze ends (BB width > 50th percentile)
            if (close[i] < ma_20[i] or bb_width_pct[i] > 50.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle band OR squeeze ends (BB width > 50th percentile)
            if (close[i] > ma_20[i] or bb_width_pct[i] > 50.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals