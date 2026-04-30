#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume regime filter and 1w trend alignment
# Bollinger Band squeeze (low volatility) precedes explosive moves in both bull and bear markets
# 1d volume regime (high/low volume percentiles) filters for institutional participation
# 1w EMA50 trend filter ensures alignment with major trend to avoid counter-trend whipsaws
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via breakouts above upper BB and bear markets via breakdowns below lower BB with trend filter.

name = "6h_BBand_Squeeze_1dVolRegime_1wEMA50_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 6h Bollinger Bands (20, 2.0)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Bollinger Band squeeze: width < 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Calculate 1d volume regime: volume > 80th percentile (high volume) OR < 20th percentile (low volume)
    vol_1d = df_1d['volume'].values
    vol_1d_series = pd.Series(vol_1d)
    vol_high_thresh = vol_1d_series.rolling(window=50, min_periods=50).quantile(0.80).values
    vol_low_thresh = vol_1d_series.rolling(window=50, min_periods=50).quantile(0.20).values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    vol_high_regime = vol_1d_aligned > vol_high_thresh
    vol_low_regime = vol_1d_aligned < vol_low_thresh
    vol_regime = vol_high_regime | vol_low_regime  # Either high or low volume regime
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend direction: price above/below EMA50
    trend_up = close_1w > ema_50_1w_aligned  # Using 1w close for trend determination
    trend_down = close_1w < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50)  # warmup for BBands and volume regimes
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(bb_middle[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_high_thresh[i]) or np.isnan(vol_low_thresh[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_bb_squeeze = bb_squeeze[i]
        curr_vol_regime = vol_regime[i]
        curr_trend_up = trend_up[i] if i < len(trend_up) else False
        curr_trend_down = trend_down[i] if i < len(trend_down) else False
        
        if position == 0:  # Flat - look for new entries
            # Require BB squeeze release and volume regime
            if not curr_bb_squeeze and curr_vol_regime:
                # Bullish entry: break above upper BB with 1w uptrend
                if curr_close > curr_bb_upper and curr_trend_up:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below lower BB with 1w downtrend
                elif curr_close < curr_bb_lower and curr_trend_down:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price returns to middle BB OR trend reverses
            if curr_close < bb_middle[i] or not curr_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price returns to middle BB OR trend reverses
            if curr_close > bb_middle[i] or not curr_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals