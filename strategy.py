#!/usr/bin/env python3
name = "4h_TRIX_Volume_Spike_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX on 4h: triple smoothed EMA of ROC
    # ROC: (close - close[period]) / close[period]
    period = 12
    roc = np.zeros(n)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period]
    
    # Triple EMA smoothing
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # 1d trend filter: EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~24 hours for 4h to reduce trades
    
    start_idx = max(100, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: TRIX crosses above zero in uptrend with volume spike
            if trix[i] > 0 and trix[i-1] <= 0 and trend_up[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: TRIX crosses below zero in downtrend with volume spike
            elif trix[i] < 0 and trix[i-1] >= 0 and trend_down[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: TRIX crosses below zero or trend changes
            if trix[i] < 0 or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or trend changes
            if trix[i] > 0 or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX zero-cross with 1d EMA34 trend filter and volume spike (2.0x) captures momentum shifts with institutional backing.
# TRIX is less noisy than MACD and provides early reversal signals. Volume spike confirms participation.
# Cooldown of 6 bars (24h) reduces trade frequency to target 15-25 trades/year.
# Works in bull markets (riding uptrend with TRIX > 0) and bear markets (riding downtrend with TRIX < 0).
# Position size 0.25 manages drawdown during volatile periods. Targets 50-100 total trades over 4 years.