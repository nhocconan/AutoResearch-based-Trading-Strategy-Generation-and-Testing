#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout + 1d Volume Regime + 1w Trend Filter
# Bollinger Band squeeze (low volatility) precedes explosive moves. Breakout direction 
# filtered by 1d volume regime (high/low volume environment) and 1w EMA50 trend.
# Works in bull/bear markets by capturing volatility expansion after consolidation.
# Target: 50-150 trades over 4 years (12-37/year) on 6h.

name = "6h_BBSqueeze_1dVolRegime_1wEMA50"
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
    
    # Calculate 1d volume regime: ratio of current volume to 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.where(vol_ma_20 > 0, vol_1d / vol_ma_20, 1.0)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Bollinger Bands (20, 2) on 6h
    if len(close) < 20:
        return np.zeros(n)
    
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + (2 * std_20)
    lower_bb = ma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / ma_20  # Normalized width
    
    # Bollinger Band Squeeze: width below 20-period mean width
    mean_width_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < mean_width_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)  # 1w EMA50 and BB warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(ma_20[i]) or np.isnan(std_20[i]) or np.isnan(mean_width_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bollinger Band breakout above upper band + 
            #            1d volume expansion (ratio > 1.5) + 
            #            1w uptrend (price > EMA50)
            if (close[i] > upper_bb[i] and 
                vol_ratio_1d_aligned[i] > 1.5 and 
                close[i] > ema_50_1w_aligned[i] and 
                squeeze[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bollinger Band breakout below lower band + 
            #             1d volume expansion (ratio > 1.5) + 
            #             1w downtrend (price < EMA50)
            elif (close[i] < lower_bb[i] and 
                  vol_ratio_1d_aligned[i] > 1.5 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  squeeze[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price re-enters Bollinger Bands (mean reversion) OR 
            #       1d volume contraction (ratio < 0.7) suggesting weak momentum
            if (close[i] < ma_20[i] or 
                vol_ratio_1d_aligned[i] < 0.7):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price re-enters Bollinger Bands (mean reversion) OR 
            #       1d volume contraction (ratio < 0.7) suggesting weak momentum
            if (close[i] > ma_20[i] or 
                vol_ratio_1d_aligned[i] < 0.7):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals