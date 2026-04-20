#!/usr/bin/env python3
"""
1d_RSI_Extreme_Trend_Filter_v1
Concept: Daily RSI extremes with weekly EMA trend filter and volume confirmation for mean-reversion entries.
- Long: RSI < 30 AND close > weekly EMA50 AND volume > 1.5x average volume
- Short: RSI > 70 AND close < weekly EMA50 AND volume > 1.5x average volume
- Exit: RSI crosses back to neutral zone (40-60)
- Position sizing: 0.25
- Target: 10-25 trades/year (40-100 total over 4 years)
- Works in bull/bear: Weekly EMA defines trend, RSI extremes capture reversals, volume confirms momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_RSI_Extreme_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Daily: RSI for mean reversion signals ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Daily: Volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Weekly: EMA50 trend filter ===
    weekly_close = df_1w['close'].values
    ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        rsi_val = rsi[i]
        close_val = close[i]
        ema50_val = ema50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(ema50_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold AND price above weekly EMA50 AND volume confirmation
            rsi_oversold = rsi_val < 30
            trend_up = close_val > ema50_val
            vol_confirm = vol_ratio_val > 1.5
            
            if rsi_oversold and trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought AND price below weekly EMA50 AND volume confirmation
            elif rsi_val > 70 and close_val < ema50_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 40 (exiting oversold zone)
            if rsi_val > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 60 (exiting overbought zone)
            if rsi_val < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals