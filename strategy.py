# 6h_1d_camarilla_breakout_volume
# Uses daily Camarilla pivot levels for breakout detection on 6h timeframe.
# Long: price breaks above R4 with volume confirmation.
# Short: price breaks below S4 with volume confirmation.
# Exit: price returns to Pivot Point level.
# This strategy targets institutional levels that work in both bull and bear markets.
# Focus on breakouts at extreme Camarilla levels (R4/S4) to avoid whipsaws.
# Volume confirmation ensures institutional participation.
# Target: 20-50 trades per year to minimize fee drag.

name = "6h_1d_camarilla_breakout_volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    typical = (high + low + close) / 3
    range_ = high - low
    
    # Camarilla levels
    r4 = close + range_ * 1.1 / 2
    r3 = close + range_ * 1.1 / 4
    r2 = close + range_ * 1.1 / 6
    r1 = close + range_ * 1.1 / 12
    pp = typical
    s1 = close - range_ * 1.1 / 12
    s2 = close - range_ * 1.1 / 6
    s3 = close - range_ * 1.1 / 4
    s4 = close - range_ * 1.1 / 2
    
    return r4, r3, r2, r1, pp, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    r4_1d = np.full_like(close_1d, np.nan)
    r3_1d = np.full_like(close_1d, np.nan)
    r2_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    pp_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    s2_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = calculate_camarilla(
            high_1d[i], low_1d[i], close_1d[i]
        )
        r4_1d[i] = r4
        r3_1d[i] = r3
        r2_1d[i] = r2
        r1_1d[i] = r1
        pp_1d[i] = pp
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    pp_6h = align_htf_to_ltf(prices, df_1d, pp_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(pp_6h[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above R4 with volume confirmation
        if close[i] > r4_6h[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below S4 with volume confirmation
        elif close[i] < s4_6h[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to pivot point level (mean reversion)
        elif position == 1 and close[i] <= pp_6h[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= pp_6h[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals