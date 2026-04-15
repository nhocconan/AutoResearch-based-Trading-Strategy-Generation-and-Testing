#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) after low BB width (<10th percentile) + 1d close > EMA50 + volume > 1.5x avg
# Short when price breaks below lower BB(20,2) after low BB width + 1d close < EMA50 + volume > 1.5x avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-25/year).
# Bollinger squeeze identifies low volatility compression before expansion. Trend filter ensures breakout alignment.
# Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend) by requiring 1d EMA50 filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: EMA50 (trend filter) ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 6h Indicators: Bollinger Bands (20,2) ===
    bb_window = 20
    bb_std = 2
    ma = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    upper_bb = ma + (bb_std * bb_std_dev)
    lower_bb = ma - (bb_std * bb_std_dev)
    bb_width = ((upper_bb - lower_bb) / ma) * 100  # percentage
    
    # BB width percentile (20-period lookback for squeeze detection)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=1).rank(pct=True).values * 100
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_window, 50) + 20  # BB(20) + EMA50(1d) + volume(20) + percentile lookback
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma_20[i]) or
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width below 10th percentile (low volatility)
        is_squeeze = bb_width_percentile[i] < 10
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper BB after squeeze
        # 2. 1d close > EMA50 (uptrend)
        # 3. Volume confirmation
        if (close[i] > upper_bb[i]) and is_squeeze and \
           (close[i] > ema50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower BB after squeeze
        # 2. 1d close < EMA50 (downtrend)
        # 3. Volume confirmation
        elif (close[i] < lower_bb[i]) and is_squeeze and \
             (close[i] < ema50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_BB_Squeeze_Breakout_1dEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0