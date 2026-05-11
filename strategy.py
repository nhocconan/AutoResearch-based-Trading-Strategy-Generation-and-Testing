#!/usr/bin/env python3
# 12h_1dCamarilla_Squeeze_Breakout
# Hypothesis: On 12h chart, trade Camarilla pivot breakouts (H3/L3) with 1d volume confirmation and squeeze filter.
# Long: price breaks above H3 + 1d volume > 1.5x 20-period avg + Bollinger Bandwidth < 50th percentile (squeeze).
# Short: price breaks below L3 + 1d volume > 1.5x 20-period avg + Bollinger Bandwidth < 50th percentile.
# Exit: price returns to Camarilla midpoint (P) or squeeze releases (BW > 50th percentile).
# Works in bull/bear: breakouts capture momentum; squeeze filters false breakouts in chop.
# Uses 1d Camarilla levels (calculated from prior 1d candle) for structure, volume for conviction.
# Target: 15-30 trades/year on 12h to avoid fee drag.

name = "12h_1dCamarilla_Squeeze_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla levels (from prior day) ---
    # Typical Price = (H + L + C)/3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3, L3, P (midpoint)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_p = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        camarilla_h3[i] = close_1d[i] + (range_1d[i] * 1.1 / 6)
        camarilla_l3[i] = close_1d[i] - (range_1d[i] * 1.1 / 6)
        camarilla_p[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
    
    # --- 1d Bollinger Bandwidth (20,2) for squeeze ---
    close_1d_series = pd.Series(close_1d)
    bb_mid = close_1d_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_1d_series.rolling(window=20, min_periods=20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_dn = bb_mid - 2 * bb_std
    bb_width = (bb_up - bb_dn) / bb_mid * 100  # Percent bandwidth
    bb_width_50th = bb_width.rolling(window=50, min_periods=50).quantile(0.5)
    squeeze = bb_width < bb_width_50th  # True when in squeeze (low volatility)
    
    camarilla_h3 = camarilla_h3.values
    camarilla_l3 = camarilla_l3.values
    camarilla_p = camarilla_p.values
    squeeze = squeeze.values
    
    # --- 1d volume confirmation ---
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 1.5)  # 50% above average
    
    # Align 1d indicators to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze.astype(float))  # bool to float
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (uses prev day), BBands(20), BBwidth percentile(50), vol MA(20)
    start_idx = 20  # 12h bars; 1d data aligned via helper
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_p_aligned[i]) or
            np.isnan(squeeze_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h3_aligned[i]
        breakout_down = close[i] < camarilla_l3_aligned[i]
        
        # Squeeze and volume conditions (must both be true)
        in_squeeze = squeeze_aligned[i] > 0.5  # True if in squeeze
        vol_spike = vol_spike_1d_aligned[i] > 0.5  # True if volume spike
        
        if position == 0:
            if breakout_up and in_squeeze and vol_spike:
                # Long: break above H3 + squeeze + volume spike
                signals[i] = 0.25
                position = 1
            elif breakout_down and in_squeeze and vol_spike:
                # Short: break below L3 + squeeze + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: return to midpoint OR squeeze releases
                if close[i] < camarilla_p_aligned[i] or squeeze_aligned[i] < 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: return to midpoint OR squeeze releases
                if close[i] > camarilla_p_aligned[i] or squeeze_aligned[i] < 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals