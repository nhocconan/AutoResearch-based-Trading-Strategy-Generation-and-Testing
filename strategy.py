#!/usr/bin/env python3
"""
Experiment #032: 12h Donchian Breakout + Volume + Chop Filter Strategy

HYPOTHESIS: Uses 12h Donchian channel breakouts (20-period) confirmed by volume spike
and choppiness regime filter. Only trades in trending markets (CHOP < 38.2) to avoid
whipsaws in ranging conditions. HTF 1d trend filter ensures alignment with higher
timeframe direction. Discrete position sizing (0.25) minimizes fee churn. Designed
to work in both bull (breakouts) and bear (breakdowns) markets by trading the
breakout direction regardless of market regime, but only when volatility is
expanding (low chop = trending).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_032_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    # Choppiness Index on 1w: CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (HHV - LLV))
    # Simplified: use ATR ratio and range expansion
    tr_1w = np.zeros(len(df_1d))
    tr_1w[0] = df_1w['high'].iloc[0] - df_1w['low'].iloc[0]
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(
            df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
            abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
            abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
        )
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
    hh_14_1w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    ll_14_1w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    chop_1w = 100 * np.log10(atr_14_1w * 14 / np.log10(14)) / np.log10((hh_14_1w - ll_14_1w) + 1e-10)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_m = (donchian_h + donchian_l) / 2
    
    # === 12h Indicators: ATR(14) for stoploss and volume filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume Spike (20-period avg) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Sufficient warmup for Donchian and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- 1d Trend Filter: Only trade when price is clearly above/below EMA50 ---
        is_uptrend_1d = price > ema50_1d_aligned[i] * 1.001
        is_downtrend_1d = price < ema50_1d_aligned[i] * 0.999
        
        # --- Regime Filter: Only trade in trending markets (CHOP < 38.2) ---
        is_trending = chop_1w_aligned[i] < 38.2
        
        # --- Volume Confirmation: Spike above average ---
        vol_confirm = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > donchian_h[i-1]  # Break above previous period high
        breakout_down = price < donchian_l[i-1]  # Break below previous period low
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if trend fails or volatility contracts
                if not is_trending or not is_uptrend_1d:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if trend fails or volatility contracts
                if not is_trending or not is_downtrend_1d:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade when all conditions align: breakout + volume + trend + regime
        if is_trending and vol_confirm:
            # Long: Donchian breakout up AND 1d uptrend
            if breakout_up and is_uptrend_1d:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down AND 1d downtrend
            elif breakout_down and is_downtrend_1d:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #032: 12h Donchian Breakout + Volume + Chop Filter Strategy

HYPOTHESIS: Uses 12h Donchian channel breakouts (20-period) confirmed by volume spike
and choppiness regime filter. Only trades in trending markets (CHOP < 38.2) to avoid
whipsaws in ranging conditions. HTF 1d trend filter ensures alignment with higher
timeframe direction. Discrete position sizing (0.25) minimizes fee churn. Designed
to work in both bull (breakouts) and bear (breakdowns) markets by trading the
breakout direction regardless of market regime, but only when volatility is
expanding (low chop = trending).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_032_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    # Choppiness Index on 1w: CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (HHV - LLV))
    # Simplified: use ATR ratio and range expansion
    tr_1w = np.zeros(len(df_1d))
    tr_1w[0] = df_1w['high'].iloc[0] - df_1w['low'].iloc[0]
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(
            df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
            abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
            abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
        )
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
    hh_14_1w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    ll_14_1w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    chop_1w = 100 * np.log10(atr_14_1w * 14 / np.log10(14)) / np.log10((hh_14_1w - ll_14_1w) + 1e-10)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_m = (donchian_h + donchian_l) / 2
    
    # === 12h Indicators: ATR(14) for stoploss and volume filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume Spike (20-period avg) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Sufficient warmup for Donchian and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- 1d Trend Filter: Only trade when price is clearly above/below EMA50 ---
        is_uptrend_1d = price > ema50_1d_aligned[i] * 1.001
        is_downtrend_1d = price < ema50_1d_aligned[i] * 0.999
        
        # --- Regime Filter: Only trade in trending markets (CHOP < 38.2) ---
        is_trending = chop_1w_aligned[i] < 38.2
        
        # --- Volume Confirmation: Spike above average ---
        vol_confirm = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > donchian_h[i-1]  # Break above previous period high
        breakout_down = price < donchian_l[i-1]  # Break below previous period low
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if trend fails or volatility contracts
                if not is_trending or not is_uptrend_1d:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if trend fails or volatility contracts
                if not is_trending or not is_downtrend_1d:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade when all conditions align: breakout + volume + trend + regime
        if is_trending and vol_confirm:
            # Long: Donchian breakout up AND 1d uptrend
            if breakout_up and is_uptrend_1d:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down AND 1d downtrend
            elif breakout_down and is_downtrend_1d:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
    
    return signals