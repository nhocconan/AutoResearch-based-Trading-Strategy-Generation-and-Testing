#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ADX_Momentum_Pivot_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d ADX for trend strength ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.maximum(high_1d - low_1d,
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di14 = 100 * plus_dm14 / (tr14 + 1e-10)
    minus_di14 = 100 * minus_dm14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-10)
    adx14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx14)
    
    # === 1d Pivot Points (previous day) ===
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # R3 and S3 levels (strong support/resistance)
    r3 = pivot + range_1d * 1.1
    s3 = pivot - range_1d * 1.1
    
    # Align to 6h
    adx_aligned = adx_1d_aligned
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 6h Momentum: ROC(12) ===
    roc_period = 12
    roc = np.zeros_like(close)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / (close[:-roc_period] + 1e-10)
    
    # === 6h Volume: current > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, roc_period, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(roc[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Strong trend filter: ADX > 25
            strong_trend = adx_aligned[i] > 25
            
            if strong_trend:
                # Momentum continuation in strong trend
                long_cond = (roc[i] > 0.005 and  # positive momentum
                            close[i] > r3_6h[i] and
                            volume[i] > vol_ma20[i])
                
                short_cond = (roc[i] < -0.005 and  # negative momentum
                             close[i] < s3_6h[i] and
                             volume[i] > vol_ma20[i])
            else:
                # Weak trend or ranging: fade at S3/R3
                long_cond = (roc[i] < -0.002 and  # recent weakness
                            close[i] < s3_6h[i] and
                            volume[i] > vol_ma20[i])
                
                short_cond = (roc[i] > 0.002 and  # recent strength
                             close[i] > r3_6h[i] and
                             volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum fade or trend weakness
            exit_cond = (roc[i] < -0.003 or  # negative momentum
                        adx_aligned[i] < 20 or  # trend weakening
                        close[i] < s3_6h[i])  # break below S3
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: momentum fade or trend weakness
            exit_cond = (roc[i] > 0.003 or  # positive momentum
                        adx_aligned[i] < 20 or  # trend weakening
                        close[i] > r3_6h[i])  # break above R3
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines 1d ADX for trend strength with 1d Pivot S3/R3 levels and 6h momentum.
# In strong trends (ADX>25): momentum breakouts at S3/R3 with volume confirmation.
# In weak trends: mean reversion fade at S3/R3. Works in bull (trend following) and
# bear (mean reversion in ranges) markets. Targets 50-150 trades over 4 years.
# Uses discrete sizing (0.25) to minimize fee churn. ADX filter reduces whipsaws.