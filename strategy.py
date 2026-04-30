#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter (EMA50) and volume confirmation (>1.8x 20-bar avg).
# Uses Bollinger Band width percentile to detect low volatility squeeze (BBW < 20th percentile).
# Breakout direction filtered by 1d EMA50 trend. Volume confirmation reduces false breakouts.
# Session filter (08-20 UTC) avoids low-liquidity periods. Discrete position sizing at ±0.25.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag on 6h.
# Works in bull markets via trend-aligned breakouts and in bear markets via mean-reversion when price re-enters BB.

name = "6h_BollingerSqueeze_Breakout_1dEMA50_Trend_VolumeConfirm_Session_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2) on 6h for squeeze detection
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + (2.0 * dev)
    lower_band = basis - (2.0 * dev)
    bb_width = (upper_band - lower_band) / basis  # Normalized width
    
    # Bollinger Band width percentile (lookback 50 periods) to detect squeeze
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile = bb_width_s.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA50 and BB calculations
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bb_width_percentile[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_bb_percentile = bb_width_percentile[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze breakout up, price > EMA50, volume spike, in session
            if (curr_close > upper_band[i] and 
                curr_bb_percentile < 20 and  # Squeeze condition: BBW in lowest 20%
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout down, price < EMA50, volume spike, in session
            elif (curr_close < lower_band[i] and 
                  curr_bb_percentile < 20 and  # Squeeze condition: BBW in lowest 20%
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price re-enters Bollinger Bands (below upper band)
            if curr_close < upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price re-enters Bollinger Bands (above lower band)
            if curr_close > lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals