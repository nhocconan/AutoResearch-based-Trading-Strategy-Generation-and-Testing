#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) after BB width < 10th percentile (squeeze) + 1d EMA50 uptrend + volume > 1.5x avg
# Short when price breaks below lower BB(20,2) after BB width < 10th percentile (squeeze) + 1d EMA50 downtrend + volume > 1.5x avg
# Uses Bollinger squeeze to identify low volatility periods primed for breakout. Works in both bull/bear by requiring 1d EMA50 alignment.
# Designed for low trade frequency (12-30/year) to minimize fee drag.

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
    
    # === 1d Indicator: EMA50 (trend filter) ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators: Bollinger Bands (20,2) and BB Width ===
    bb_window = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    std_20 = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    upper_bb = sma_20 + (std_20 * bb_std)
    lower_bb = sma_20 - (std_20 * bb_std)
    bb_width = (upper_bb - lower_bb) / sma_20 * 100  # as percentage
    
    # BB Width percentile for squeeze detection (10th percentile lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50, raw=False
    ).values
    # Simplified: squeeze when BB width < rolling 10th percentile
    squeeze_threshold = pd.Series(bb_width).rolling(window=50, min_periods=20).quantile(0.10).values
    is_squeeze = bb_width < squeeze_threshold
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_window, 50) + 20  # BB(20) + EMA50(1d) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i]) or
            np.isnan(squeeze_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper BB after squeeze
        # 2. 1d EMA50 uptrend (price > EMA50)
        # 3. Volume confirmation
        if (close[i] > upper_bb[i]) and \
           (is_squeeze[i-1] if i > 0 else False) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower BB after squeeze
        # 2. 1d EMA50 downtrend (price < EMA50)
        # 3. Volume confirmation
        elif (close[i] < lower_bb[i]) and \
             (is_squeeze[i-1] if i > 0 else False) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_BB_Squeeze_Breakout_1dEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0