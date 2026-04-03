#!/usr/bin/env python3
"""
Experiment #370: 1d Donchian(20) Breakout + Weekly Volume Spike + Weekly ATR Filter

HYPOTHESIS: Daily Donchian(20) breakouts capture institutional participation when confirmed by 
weekly volume spikes (>2.0x average) and sufficient volatility (weekly ATR > 20-day mean ATR). 
This combines price structure (Donchian channels) with volume confirmation and volatility 
regime filtering to avoid choppy markets. Targets 15-25 trades/year on 1d timeframe 
(60-100 total over 4 years) for minimal fee drag. Works in both bull (breakouts continue) 
and bear (breakdowns continue) markets via symmetric long/short logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_weekly_vol_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume spike and ATR filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on weekly
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0  # Neutral for warmup
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # Calculate ATR(14) on weekly
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = high_1w[1:] - low_1w[1:]
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.concatenate([[high_1w[0] - low_1w[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        atr_1w = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        # 20-period mean of weekly ATR for regime filter
        atr_ma_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
        atr_ratio_1w = np.zeros(len(atr_1w))
        atr_ratio_1w[20:] = atr_1w[20:] / atr_ma_20[20:]
        atr_ratio_1w[:20] = 1.0  # Neutral for warmup
        atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    else:
        atr_ratio_1w_aligned = np.full(n, 1.0)
    
    # === Calculate Daily Donchian(20) channels ===
    # Use expanding window for warmup, then rolling 20
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            # Expanding window for warmup period
            donchian_h[i] = np.max(high[:i+1])
            donchian_l[i] = np.min(low[:i+1])
        else:
            # Rolling 20-period window
            donchian_h[i] = np.max(high[i-19:i+1])
            donchian_l[i] = np.min(low[i-19:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    warmup = 60  # Ensure enough data for HTF and Donchian calculation
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or
            np.isnan(vol_ratio_1w_aligned[i]) or np.isnan(atr_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filters ---
        # Volume confirmation: weekly volume spike (> 2.0x average)
        volume_spike = vol_ratio_1w_aligned[i] > 2.0
        
        # Volatility regime: weekly ATR > 20-day mean ATR (avoid low volatility chop)
        vol_regime = atr_ratio_1w_aligned[i] > 1.0
        
        # Only trade when both filters are present
        trade_allowed = volume_spike and vol_regime
        
        # --- Donchian Breakout Logic ---
        long_breakout = close[i] > donchian_h[i]
        short_breakout = close[i] < donchian_l[i]
        
        # --- Entry Logic (Only if Flat) ---
        if long_breakout and trade_allowed:
            signals[i] = SIZE
        elif short_breakout and trade_allowed:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #370: 1d Donchian(20) Breakout + Weekly Volume Spike + Weekly ATR Filter

HYPOTHESIS: Daily Donchian(20) breakouts capture institutional participation when confirmed by 
weekly volume spikes (>2.0x average) and sufficient volatility (weekly ATR > 20-day mean ATR). 
This combines price structure (Donchian channels) with volume confirmation and volatility 
regime filtering to avoid choppy markets. Targets 15-25 trades/year on 1d timeframe 
(60-100 total over 4 years) for minimal fee drag. Works in both bull (breakouts continue) 
and bear (breakdowns continue) markets via symmetric long/short logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_weekly_vol_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume spike and ATR filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on weekly
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0  # Neutral for warmup
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # Calculate ATR(14) on weekly
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = high_1w[1:] - low_1w[1:]
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.concatenate([[high_1w[0] - low_1w[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        atr_1w = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        # 20-period mean of weekly ATR for regime filter
        atr_ma_20 = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
        atr_ratio_1w = np.zeros(len(atr_1w))
        atr_ratio_1w[20:] = atr_1w[20:] / atr_ma_20[20:]
        atr_ratio_1w[:20] = 1.0  # Neutral for warmup
        atr_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    else:
        atr_ratio_1w_aligned = np.full(n, 1.0)
    
    # === Calculate Daily Donchian(20) channels ===
    # Use expanding window for warmup, then rolling 20
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            # Expanding window for warmup period
            donchian_h[i] = np.max(high[:i+1])
            donchian_l[i] = np.min(low[:i+1])
        else:
            # Rolling 20-period window
            donchian_h[i] = np.max(high[i-19:i+1])
            donchian_l[i] = np.min(low[i-19:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    warmup = 60  # Ensure enough data for HTF and Donchian calculation
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or
            np.isnan(vol_ratio_1w_aligned[i]) or np.isnan(atr_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filters ---
        # Volume confirmation: weekly volume spike (> 2.0x average)
        volume_spike = vol_ratio_1w_aligned[i] > 2.0
        
        # Volatility regime: weekly ATR > 20-day mean ATR (avoid low volatility chop)
        vol_regime = atr_ratio_1w_aligned[i] > 1.0
        
        # Only trade when both filters are present
        trade_allowed = volume_spike and vol_regime
        
        # --- Donchian Breakout Logic ---
        long_breakout = close[i] > donchian_h[i]
        short_breakout = close[i] < donchian_l[i]
        
        # --- Entry Logic (Only if Flat) ---
        if long_breakout and trade_allowed:
            signals[i] = SIZE
        elif short_breakout and trade_allowed:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals