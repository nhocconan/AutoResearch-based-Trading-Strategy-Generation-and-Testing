#!/usr/bin/env python3
# 1D_1W_Camarilla_R1_S1_Breakout_Trend_Follow_v2
# Hypothesis: Trade Camarilla pivot breakouts on 1d with 1w trend filter and volume confirmation.
# Long when: price breaks above R1 on 1d, 1w trend is up (close > EMA50), volume > 1.5x average.
# Short when: price breaks below S1 on 1d, 1w trend is down (close < EMA50), volume > 1.5x average.
# Uses volatility filter: only trade when 1d ATR(14) > 0.5 * ATR(50) to avoid choppy markets.
# Works in bull/bear by following weekly trend and using Camarilla levels for precise entries.
# Target: 10-25 trades/year per symbol.

name = "1D_1W_Camarilla_R1_S1_Breakout_Trend_Follow_v2"
timeframe = "1d"
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
    
    # 1d indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Calculate Camarilla levels for previous day
    # Need previous day's high, low, close
    prev_high = high_s.shift(1)
    prev_low = low_s.shift(1)
    prev_close = close_s.shift(1)
    
    # Typical price for pivot calculation
    pp = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r1 = pp + (range_val * 1.1 / 12)
    s1 = pp - (range_val * 1.1 / 12)
    
    # ATR for volatility filter
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    atr50 = tr.rolling(window=50, min_periods=50).mean()
    vol_filter = (atr14 > 0.5 * atr50).values  # Avoid choppy markets
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to 1d
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + price breaks above R1 + volume + volatility filter
            if weekly_up and volume_confirm and vol_filter[i]:
                if close[i] > r1[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: weekly downtrend + price breaks below S1 + volume + volatility filter
            elif weekly_down and volume_confirm and vol_filter[i]:
                if close[i] < s1[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions: weekly trend changes or price breaks below S1 (mean reversion)
            if not weekly_up or close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: weekly trend changes or price breaks above R1 (mean reversion)
            if not weekly_down or close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals