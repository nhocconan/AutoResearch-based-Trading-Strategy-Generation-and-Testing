#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(14) mean reversion with Bollinger Bands and volume confirmation
# Uses 1d EMA50 trend filter and 1d volume spike to confirm mean reversion setups
# Designed for 20-50 trades/year with proper risk control via trend failure
# RSI < 30 for long, RSI > 70 for short, with price near BB bands
# Volume filter requires current 1d volume > 1.5x 20-day average
# Trend filter: price > 1d EMA50 for long, price < 1d EMA50 for short
name = "4h_RSI_MeanReversion_BB_VolumeTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-day average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Bollinger Bands (20, 2) on 4h data
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Calculate RSI(14) on 4h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-day average
        # Find the most recent completed 1d bar
        idx_1d = len(df_1d) - 1
        while idx_1d >= 0 and df_1d.iloc[idx_1d]['open_time'] > prices.iloc[i]['open_time']:
            idx_1d -= 1
        vol_filter = False
        if idx_1d >= 0:
            vol_1d_current = df_1d.iloc[idx_1d]['volume']
            vol_filter = vol_1d_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for mean reversion setup
            # Long: RSI < 30 and price near lower BB
            if rsi[i] < 30 and close[i] <= bb_lower[i] * 1.02 and ema50_aligned[i] > 0:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: RSI > 70 and price near upper BB
            elif rsi[i] > 70 and close[i] >= bb_upper[i] * 0.98 and ema50_aligned[i] < 0:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: RSI > 50 or price touches middle BB or trend fails
            if rsi[i] > 50 or close[i] >= bb_middle[i] or ema50_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 50 or price touches middle BB or trend fails
            if rsi[i] < 50 or close[i] <= bb_middle[i] or ema50_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals