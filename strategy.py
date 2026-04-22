#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and 1w volume confirmation
# Williams %R (14) identifies overbought/oversold conditions.
# Trend filter: 1d EMA50 (bullish if close > EMA50, bearish if close < EMA50).
# Volume confirmation: 1w volume > 1.5x 4-week average to avoid false signals.
# Works in bull markets by buying oversold dips in uptrend and in bear markets by selling overbought rallies in downtrend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Load 1w data for volume confirmation (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 4:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Williams %R (14) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w volume 4-period average for spike detection
    vol_avg_4_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    vol_avg_4_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_4_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_avg_4_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + 1d uptrend + 1w volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_4_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + 1d downtrend + 1w volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_4_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to -50 (mean reversion) or trend reversal
            if position == 1:
                # Exit on return to -50 or trend reversal to downtrend
                if (williams_r[i] >= -50 or 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on return to -50 or trend reversal to uptrend
                if (williams_r[i] <= -50 or 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50_1wVolSpike"
timeframe = "6h"
leverage = 1.0