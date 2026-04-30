#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1w EMA200 trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) + 1w EMA200 uptrend + volume > 2x 20-bar average.
# Short when price breaks below lower BB(20,2) + 1w EMA200 downtrend + volume > 2x 20-bar average.
# Bollinger squeeze identifies low volatility breakouts, effective in both bull and bear markets.
# 1w EMA200 filter ensures alignment with higher-timeframe trend, improving performance.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# Targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.25).

name = "6h_Bollinger_Squeeze_Breakout_1wEMA200_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    
    # Bollinger Band Width for squeeze detection (percentile lookback)
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=20).rank(pct=True).values
    is_squeeze = bb_width_percentile < 0.2  # Bottom 20% = squeeze
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(bb_width_percentile[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        is_uptrend = curr_close > ema_200_aligned[i]
        is_downtrend = curr_close < ema_200_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze breakout above upper band + uptrend + volume confirmation
            if (is_squeeze[i-1] and curr_close > upper_bb[i] and 
                is_uptrend and curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower band + downtrend + volume confirmation
            elif (is_squeeze[i-1] and curr_close < lower_bb[i] and 
                  is_downtrend and curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when price closes below middle Bollinger Band (mean reversion)
            if curr_close < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above middle Bollinger Band (mean reversion)
            if curr_close > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals