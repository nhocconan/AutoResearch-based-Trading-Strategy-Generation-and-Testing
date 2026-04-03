#!/usr/bin/env python3
"""
Experiment #1047: 6h Elder Ray + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) measures buying/selling pressure relative to EMA13.
Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND weekly pivot shows uptrend (price > weekly pivot) AND volume spike (>1.5x avg).
Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND weekly pivot shows downtrend (price < weekly pivot) AND volume spike.
Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown. Target: 75-150 total trades over 4 years (19-38/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1047_6h_elder_ray_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA13 (used in Elder Ray) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # === HTF: 1w data for weekly pivot ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray ===
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = 20  # sufficient for EMA and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema13_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Entry Conditions ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Elder Ray: Bull Power > 0 AND Bear Power < 0 = bullish momentum
            # Elder Ray: Bear Power < 0 AND Bull Power < 0 = bearish momentum
            bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0
            bearish_momentum = bear_power[i] < 0 and bull_power[i] < 0
            
            # Weekly pivot direction: price > pivot = uptrend, price < pivot = downtrend
            pivot_uptrend = price > weekly_pivot_aligned[i]
            pivot_downtrend = price < weekly_pivot_aligned[i]
            
            # Long: bullish momentum AND weekly pivot uptrend
            if bullish_momentum and pivot_uptrend:
                signals[i] = SIZE
            # Short: bearish momentum AND weekly pivot downtrend
            elif bearish_momentum and pivot_downtrend:
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
"""
Experiment #1047: 6h Elder Ray + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) measures buying/selling pressure relative to EMA13.
Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND weekly pivot shows uptrend (price > weekly pivot) AND volume spike (>1.5x avg).
Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND weekly pivot shows downtrend (price < weekly pivot) AND volume spike.
Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown. Target: 75-150 total trades over 4 years (19-38/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1047_6h_elder_ray_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA13 (used in Elder Ray) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # === HTF: 1w data for weekly pivot ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === 6h Indicators: Elder Ray ===
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = 20  # sufficient for EMA and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema13_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Entry Conditions ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Elder Ray: Bull Power > 0 AND Bear Power < 0 = bullish momentum
            # Elder Ray: Bear Power < 0 AND Bull Power < 0 = bearish momentum
            bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0
            bearish_momentum = bear_power[i] < 0 and bull_power[i] < 0
            
            # Weekly pivot direction: price > pivot = uptrend, price < pivot = downtrend
            pivot_uptrend = price > weekly_pivot_aligned[i]
            pivot_downtrend = price < weekly_pivot_aligned[i]
            
            # Long: bullish momentum AND weekly pivot uptrend
            if bullish_momentum and pivot_uptrend:
                signals[i] = SIZE
            # Short: bearish momentum AND weekly pivot downtrend
            elif bearish_momentum and pivot_downtrend:
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals