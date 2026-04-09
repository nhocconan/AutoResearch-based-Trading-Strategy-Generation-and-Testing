#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h breakout strategy using 4h Camarilla pivot + 1d trend filter + volume confirmation + session (08-20 UTC)
# Uses 4h for structure (Camarilla H3/L3 breakouts), 1d for trend bias (price > EMA50 = long bias, < EMA50 = short bias)
# Enters only in direction of 1d trend during active session with volume spike (>2.0x 20-period MA)
# Discrete position sizing 0.20 targets ~15-25 trades/year to minimize fee drag
# Works in bull/bear: trend filter avoids counter-trend breakouts, session filter avoids low-liquidity periods

name = "1h_4h_1d_camarilla_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (based on prior 4h bar)
    camarilla_h3 = close_4h + 1.25 * (high_4h - low_4h)
    camarilla_l3 = close_4h - 1.25 * (high_4h - low_4h)
    
    # Align 4h Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50 for trend bias
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC) and volume confirmation
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 2.0 * vol_ma_20  # Pre-compute boolean array
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price falls below L3 level
            if close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit short if price rises above H3 level
            if close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Breakout strategy: enter only in direction of 1d trend with volume confirmation
            if close[i] > camarilla_h3_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_confirmed[i]:
                position = 1
                signals[i] = 0.20
            elif close[i] < camarilla_l3_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_confirmed[i]:
                position = -1
                signals[i] = -0.20
    
    return signals