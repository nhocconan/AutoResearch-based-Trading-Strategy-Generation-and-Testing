#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeSpike_Regime_New
Hypothesis: 4h timeframe strategy using Camarilla R1/S1 breakouts filtered by 12h EMA50 trend and volume spikes. Targets 75-200 trades over 4 years (19-50/year) with 0.25 position size. Uses Bollinger Bandwidth regime filter to avoid whipsaws in low volatility and capture trends. Designed to work in both bull and bear markets by aligning with the 12h trend and using volume confirmation for momentum validation. Uses tighter volume confirmation (2.5x) and adjusted regime threshold to reduce overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Get 1d data for Camarilla R1/S1 levels (from previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d data for Camarilla R1/S1 levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.08)   # R1 level
    s1 = prev_close - (rng * 1.08)   # S1 level
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.5 * 20-period average (tighter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.5 * vol_avg)
    
    # Bollinger Bandwidth regime filter (20-period, 2 std dev) on 4h
    close_series = pd.Series(close)
    ma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + (2 * std_20)
    lower_bb = ma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / ma_20
    # Regime: avoid extremely low volatility (choppy sideways) - use BW > 10th percentile (tighter)
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    # Only trade when volatility is above extreme lows (percentile > 0.1)
    volatility_regime = bb_width_percentile > 0.1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need 12h EMA50 (50), 1d shift(1) for Camarilla, vol avg (20), BB (100 for percentile)
    start_idx = max(50 + 1, 1 + 1, 20, 100)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(volatility_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        vol_reg = volatility_regime[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 12h EMA50 alignment, volume confirmation, and volatility regime
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_conf and 
                            vol_reg)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_conf and 
                             vol_reg)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 12h EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_VolumeSpike_Regime_New"
timeframe = "4h"
leverage = 1.0