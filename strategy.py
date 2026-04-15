#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Long when price breaks above upper BB (20,2) after squeeze (BW < 20th percentile) + 1d EMA50 uptrend + volume > 1.5x avg
# Short when price breaks below lower BB (20,2) after squeeze + 1d EMA50 downtrend + volume > 1.5x avg
# Uses Bollinger Band width percentile to identify low volatility squeezes that precede explosive moves.
# Works in bull markets (breakouts continuation) and bear markets (panic selling exhaustion) by requiring 1d EMA50 alignment.
# Designed for low trade frequency (15-30/year) with discrete position sizing (0.25) to minimize fee churn.

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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 (trend filter) ===
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 6h Indicators: Bollinger Bands (20,2) and Band Width Percentile ===
    bb_window = 20
    bb_std = 2
    
    # Calculate Bollinger Bands
    sma_20 = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    std_20 = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    
    # Bollinger Band Width and its percentile rank (252-period ~ 1 year of 6h data)
    bb_width = (upper_bb - lower_bb) / sma_20
    # Calculate percentile rank of current BB width vs lookback period
    bb_width_percentile = np.full_like(bb_width, np.nan)
    lookback = 252  # ~1 year of 6h bars
    for i in range(lookback, len(bb_width)):
        if not np.isnan(bb_width[i-lookback:i+1]).any():
            bb_width_percentile[i] = (bb_width[i] > bb_width[i-lookback:i+1]).mean() * 100
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_window, 50, 20) + lookback  # BB(20) + EMA50 + percentile lookback
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_sma_20[i]) or
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: Bollinger Band Width below 20th percentile (low volatility)
        is_squeeze = bb_width_percentile[i] < 20
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper Bollinger Band (20,2)
        # 2. Bollinger Band squeeze (low volatility breakout)
        # 3. 1d EMA50 uptrend (price above EMA50)
        # 4. Volume confirmation
        if (close[i] > upper_bb[i]) and \
           is_squeeze and \
           (close[i] > ema_50_aligned[i]) and \
           vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower Bollinger Band (20,2)
        # 2. Bollinger Band squeeze (low volatility breakout)
        # 3. 1d EMA50 downtrend (price below EMA50)
        # 4. Volume confirmation
        elif (close[i] < lower_bb[i]) and \
             is_squeeze and \
             (close[i] < ema_50_aligned[i]) and \
             vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_BB_Squeeze_Breakout_1dEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0