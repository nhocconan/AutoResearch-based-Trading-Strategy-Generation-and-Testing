#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter_v2
Hypothesis: Camarilla R1/S1 breakout with 1d trend filter and choppiness regime (CHOP > 61.8 = range, < 38.2 = trend) to avoid whipsaw. Uses volume confirmation (>1.5x median) for conviction. ATR trailing stop (2.0x) for risk management. Designed for BTC/ETH with moderate trade frequency (~30-50/year) to balance edge and fees. Works in bull/bear by only trading with 1d trend direction and avoiding ranging markets via CHOP filter.
"""

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
    
    # Get 1d data for HTF trend (EMA34) and choppiness
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d Choppiness Index (CHOP) - range: 0-100, >61.8 = range, <38.2 = trend
    hl_range = pd.Series(df_1d['high'].values - df_1d['low'].values).rolling(window=14, min_periods=14).sum()
    true_range = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
            np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
        )
    )
    true_range[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    tr_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(tr_sum / hl_range) / np.log10(14)
    chop_values = chop.values
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar (HLC of prior 4h)
    cam_high = pd.Series(df_4h['high'].values).shift(1).values
    cam_low = pd.Series(df_4h['low'].values).shift(1).values
    cam_close = pd.Series(df_4h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (standard breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Volume confirmation: volume > 1.5x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(34) 1d, CHOP 1d, Camarilla (need 2 bars for shift), volume median (20), ATR (14)
    start_idx = max(34, 14, 2, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        # Volume confirmation: only trade in above-average volume
        volume_confirm = volume_val > 1.5 * vol_median_val
        
        if position == 0:
            # Long: break above R1 with volume confirmation, uptrend, and trending regime
            long_signal = (close_val > r1_val) and \
                          volume_confirm and \
                          uptrend and \
                          trending_regime
            
            # Short: break below S1 with volume confirmation, downtrend, and trending regime
            short_signal = (close_val < s1_val) and \
                           volume_confirm and \
                           downtrend and \
                           trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter_v2"
timeframe = "4h"
leverage = 1.0