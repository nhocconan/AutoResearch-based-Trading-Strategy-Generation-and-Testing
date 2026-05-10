#!/usr/bin/env python3
"""
1d_WickReversal_VolumeSpike
Hypothesis: Captures reversals at long-wick rejections of daily highs/lows with volume confirmation.
Works in bull markets by buying dips rejected at daily lows and in bear markets by selling rallies rejected at daily highs.
Uses 1d timeframe for signal generation and 1w trend filter. Target: 15-25 trades/year per symbol.
"""

name = "1d_WickReversal_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Convert to Series for indicator calculations
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # Daily body and wick calculations
    body = np.abs(close - open_price)
    total_range = high - low
    lower_wick = np.minimum(open_price, close) - low
    upper_wick = high - np.maximum(open_price, close)
    
    # Wick ratio (wick as % of total range) - avoids division by zero
    lower_wick_ratio = np.divide(lower_wick, total_range, out=np.zeros_like(lower_wick), where=total_range!=0)
    upper_wick_ratio = np.divide(upper_wick, total_range, out=np.zeros_like(upper_wick), where=total_range!=0)
    
    # Volume confirmation (20-period average)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1w trend filter: EMA34 on weekly closes
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align 1w trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lower_wick_ratio[i]) or np.isnan(upper_wick_ratio[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        # Conditions for long entry: bullish rejection at low
        long_wick_rejection = lower_wick_ratio[i] > 0.6
        not_overbought = upper_wick_ratio[i] < 0.3
        
        # Conditions for short entry: bearish rejection at high
        short_wick_rejection = upper_wick_ratio[i] > 0.6
        not_oversold = lower_wick_ratio[i] < 0.3
        
        if position == 0:
            # Enter long: bullish rejection + 1w uptrend + volume
            if (long_wick_rejection and not_overbought and
                trend_1w_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish rejection + 1w downtrend + volume
            elif (short_wick_rejection and not_oversold and
                  trend_1w_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when rejection fails or trend changes
            if (lower_wick_ratio[i] < 0.3 or trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when rejection fails or trend changes
            if (upper_wick_ratio[i] < 0.3 or trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals