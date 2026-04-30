#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation.
# Uses Bollinger Band Width (BBW) percentile to detect low volatility squeeze (breakout precursor).
# 1d EMA50 for higher timeframe trend direction filter.
# Volume confirmation (>1.5x 20-bar avg) to reduce false breakouts.
# Session filter (08-20 UTC) to trade only during liquid hours.
# Discrete position sizing at ±0.25 to manage fee drag.
# Target: 60-120 total trades over 4 years (15-30/year) to avoid excessive fees on 6h timeframe.
# Works in bull markets via breakout continuation and in bear markets via volatility expansion capture.

name = "6h_BollingerSqueeze_Breakout_1dEMA50_VolumeConfirm_Session_v1"
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
    close_1d_vals = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2.0
    ma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = ma_20 + (bb_std * std_20)
    lower_band = ma_20 - (bb_std * std_20)
    
    # Bollinger Band Width (BBW) = (Upper - Lower) / Middle
    bbw = (upper_band - lower_band) / ma_20
    # BBW percentile rank over 50 periods to detect squeeze (low volatility)
    bbw_series = pd.Series(bbw)
    bbw_percentile = bbw_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for BBW percentile and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bbw_percentile[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_bbw_percentile = bbw_percentile[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ma_20 = ma_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: BBW squeeze breakout (low percentile), price > upper band, close > 1d EMA50, volume spike
            if (curr_bbw_percentile <= 0.2 and  # BBW in lowest 20% = squeeze
                curr_close > upper_band[i] and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: BBW squeeze breakout, price < lower band, close < 1d EMA50, volume spike
            elif (curr_bbw_percentile <= 0.2 and  # BBW in lowest 20% = squeeze
                  curr_close < lower_band[i] and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to middle band (mean reversion) or volatility expands
            if curr_close < curr_ma_20:  # Price back below 20-period MA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price returns to middle band
            if curr_close > curr_ma_20:  # Price back above 20-period MA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals