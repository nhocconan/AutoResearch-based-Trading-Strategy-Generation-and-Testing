#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width + 1d EMA Trend + Volume Spike
# Long when: BB Width < 20th percentile (low volatility squeeze) AND price > 1d EMA50 AND volume > 2x 20-period MA
# Short when: BB Width < 20th percentile (low volatility squeeze) AND price < 1d EMA50 AND volume > 2x 20-period MA
# Exit when: BB Width > 50th percentile (volatility expansion) OR EMA filter fails
# Uses BB Width squeeze for low volatility breakouts, 1d EMA for trend filter, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_BBWidth_1dEMA_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands on 6h
    if len(close) >= 20:
        sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = sma_20 + (2 * std_20)
        lower_bb = sma_20 - (2 * std_20)
        bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    else:
        sma_20 = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for spike detection
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1d = np.full(len(close_1d), np.nan)
    
    # Calculate BB Width percentiles on 6h (20th and 50th)
    bb_width_p20 = np.full(n, np.nan)
    bb_width_p50 = np.full(n, np.nan)
    if len(bb_width) >= 50:
        for i in range(49, n):
            window = bb_width[max(0, i-49):i+1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) >= 20:
                bb_width_p20[i] = np.percentile(valid_window, 20)
                bb_width_p50[i] = np.percentile(valid_window, 50)
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # BB Width squeeze conditions
    bb_squeeze = bb_width < bb_width_p20  # Low volatility squeeze
    bb_expansion = bb_width > bb_width_p50  # Volatility expansion (exit signal)
    
    # EMA trend filter
    price_above_ema = close > ema_50_1d_aligned
    price_below_ema = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(bb_width[i]) or np.isnan(bb_width_p20[i]) or np.isnan(bb_width_p50[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: BB squeeze + price above 1d EMA50 + volume spike
            if (bb_squeeze[i] and 
                price_above_ema[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: BB squeeze + price below 1d EMA50 + volume spike
            elif (bb_squeeze[i] and 
                  price_below_ema[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: BB expansion OR price below 1d EMA50
            if (bb_expansion[i] or price_below_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: BB expansion OR price above 1d EMA50
            if (bb_expansion[i] or price_above_ema[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals