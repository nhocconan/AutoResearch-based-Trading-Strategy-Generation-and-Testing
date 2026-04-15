#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) + BB width at 20-period low (squeeze) + 1d EMA50 > EMA200 (uptrend) + volume > 1.5x avg
# Short when price breaks below lower BB(20,2) + BB width at 20-period low (squeeze) + 1d EMA50 < EMA200 (downtrend) + volume > 1.5x avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-37/year).
# Bollinger squeeze identifies low volatility periods primed for breakouts. Trend filter ensures we trade in direction of higher TF trend.
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend) by requiring EMA alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # === 1d Indicator: EMA50 and EMA200 (trend filter) ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 1 = uptrend (EMA50 > EMA200), -1 = downtrend (EMA50 < EMA200), 0 = no trend
    trend_1d = np.where(ema50_1d > ema200_1d, 1, np.where(ema50_1d < ema200_1d, -1, 0))
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicator: Bollinger Bands (20,2) and BB Width ===
    bb_window = 20
    bb_std = 2
    
    # Calculate Bollinger Bands
    sma = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    std = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Calculate Bollinger Band Width (normalized)
    bb_width = (upper_band - lower_band) / sma
    
    # Calculate 20-period rolling minimum of BB Width (for squeeze detection)
    bb_width_min = pd.Series(bb_width).rolling(window=bb_window, min_periods=bb_window).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_window, 200) + 20  # BB(20) + EMA200(200) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(bb_width_min[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB Width at 20-period low
        squeeze_condition = bb_width[i] <= bb_width_min[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper Bollinger Band
        # 2. Bollinger Band squeeze (low volatility)
        # 3. Uptrend on 1d (EMA50 > EMA200)
        # 4. Volume confirmation
        if (close[i] > upper_band[i]) and \
           squeeze_condition and \
           (trend_1d_aligned[i] == 1) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower Bollinger Band
        # 2. Bollinger Band squeeze (low volatility)
        # 3. Downtrend on 1d (EMA50 < EMA200)
        # 4. Volume confirmation
        elif (close[i] < lower_band[i]) and \
             squeeze_condition and \
             (trend_1d_aligned[i] == -1) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_BB20_2_Squeeze_1dEMA50_200_Trend_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0