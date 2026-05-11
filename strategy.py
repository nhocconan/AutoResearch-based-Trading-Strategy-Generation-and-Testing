# 1D_1W_RangeBreakout_Pullback
# Hypothesis: Identifies range-bound markets on 1d using weekly Bollinger Bands and enters long/short on pullbacks to mid-Bollinger Band with volume confirmation. Works in bull markets (buying dips in uptrends) and bear markets (selling rallies in downtrends) by trading mean reversion within the weekly range. Uses weekly trend filter to avoid counter-trend trades and avoid whipsaws.

name = "1D_1W_RangeBreakout_Pullback"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for Bollinger Bands and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w Bollinger Bands (20, 2) ---
    close_1w = df_1w['close'].values
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + 2 * std_20_1w
    lower_bb_1w = sma_20_1w - 2 * std_20_1w
    middle_bb_1w = sma_20_1w  # same as SMA20
    
    # --- 1w Trend (SMA50 slope) ---
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_slope = sma_50_1w - np.roll(sma_50_1w, 1)
    sma_50_1w_slope[0] = 0
    sma_50_1w_slope = pd.Series(sma_50_1w_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    
    # Align 1w indicators to daily
    upper_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    middle_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, middle_bb_1w)
    sma_50_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w_slope)
    
    # --- Volume confirmation (volume > 20-day average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 20-day vol MA and 50-week SMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(upper_bb_1w_aligned[i]) or
            np.isnan(lower_bb_1w_aligned[i]) or
            np.isnan(middle_bb_1w_aligned[i]) or
            np.isnan(sma_50_1w_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend direction
        uptrend = sma_50_1w_slope_aligned[i] > 0
        downtrend = sma_50_1w_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long pullback to middle BB in uptrend
                if close[i] <= middle_bb_1w_aligned[i] and close[i] >= lower_bb_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short pullback to middle BB in downtrend
                if close[i] >= middle_bb_1w_aligned[i] and close[i] <= upper_bb_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price reaches upper BB or trend turns down
                if close[i] >= upper_bb_1w_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches lower BB or trend turns up
                if close[i] <= lower_bb_1w_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals