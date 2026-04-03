#!/usr/bin/env python3
"""
Experiment #262: 12h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels from 1d timeframe act as strong support/resistance zones. 
Price touching L3/H3 levels with volume spike indicates institutional interest. 
Choppiness regime filter ensures we only trade in trending markets (CHOP < 38.2) or 
transition markets (38.2 <= CHOP <= 61.8), avoiding pure ranging markets where 
pivot reversals fail. Uses discrete 0.25 position sizing to limit drawdown. 
Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    if len(df_1d) >= 2:
        # Use previous day's OHLC for today's levels (no look-ahead)
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Camarilla levels
        range_ = prev_high - prev_low
        camarilla_h3 = prev_close + range_ * 1.1 / 4
        camarilla_l3 = prev_close - range_ * 1.1 / 4
        camarilla_h4 = prev_close + range_ * 1.1 / 2
        camarilla_l4 = prev_close - range_ * 1.1 / 2
        
        # Align to 12h timeframe (shifted by 1 day already in calculation)
        h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    else:
        h3_12h = np.full(n, np.nan)
        l3_12h = np.full(n, np.nan)
        h4_12h = np.full(n, np.nan)
        l4_12h = np.full(n, np.nan)
    
    # === HTF: 1w data for choppiness regime filter ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index(14) on weekly data
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr_1w = np.zeros(len(close_1w))
        tr_1w[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
        chop_1w = np.full(len(close_1w), np.nan)
        valid = (sum_tr_14 > 0) & (max_high_14 > min_low_14) & ~(np.isnan(sum_tr_14) | np.isnan(max_high_14) | np.isnan(min_low_14))
        chop_1w[valid] = 100 * np.log10(sum_tr_14[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
        
        # Align to 12h timeframe
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Volume spike detection: volume > 2.0 * 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Avoid strong ranging markets (Choppiness > 61.8) ---
        # Only trade when market is trending (CHOP < 38.2) or transition (38.2-61.8)
        if chop_1w_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        # --- Price Action at Camarilla Levels with Volume Spike ---
        # Long: Price touches or goes below L3/L4 with volume spike
        touch_long = (low[i] <= l3_12h[i]) or (low[i] <= l4_12h[i])
        # Short: Price touches or goes above H3/H4 with volume spike
        touch_short = (high[i] >= h3_12h[i]) or (high[i] >= h4_12h[i])
        
        long_signal = touch_long and volume_spike[i]
        short_signal = touch_short and volume_spike[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using 12h data
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = close[entry_bar] - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if high[i] >= h3_12h[i]:  # Reached H3 level
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if low[i] <= l3_12h[i]:  # Reached L3 level
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if long_signal:
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #262: 12h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels from 1d timeframe act as strong support/resistance zones. 
Price touching L3/H3 levels with volume spike indicates institutional interest. 
Choppiness regime filter ensures we only trade in trending markets (CHOP < 38.2) or 
transition markets (38.2 <= CHOP <= 61.8), avoiding pure ranging markets where 
pivot reversals fail. Uses discrete 0.25 position sizing to limit drawdown. 
Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    if len(df_1d) >= 2:
        # Use previous day's OHLC for today's levels (no look-ahead)
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Camarilla levels
        range_ = prev_high - prev_low
        camarilla_h3 = prev_close + range_ * 1.1 / 4
        camarilla_l3 = prev_close - range_ * 1.1 / 4
        camarilla_h4 = prev_close + range_ * 1.1 / 2
        camarilla_l4 = prev_close - range_ * 1.1 / 2
        
        # Align to 12h timeframe (shifted by 1 day already in calculation)
        h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    else:
        h3_12h = np.full(n, np.nan)
        l3_12h = np.full(n, np.nan)
        h4_12h = np.full(n, np.nan)
        l4_12h = np.full(n, np.nan)
    
    # === HTF: 1w data for choppiness regime filter ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index(14) on weekly data
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr_1w = np.zeros(len(close_1w))
        tr_1w[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
        chop_1w = np.full(len(close_1w), np.nan)
        valid = (sum_tr_14 > 0) & (max_high_14 > min_low_14) & ~(np.isnan(sum_tr_14) | np.isnan(max_high_14) | np.isnan(min_low_14))
        chop_1w[valid] = 100 * np.log10(sum_tr_14[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
        
        # Align to 12h timeframe
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Volume spike detection: volume > 2.0 * 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Avoid strong ranging markets (Choppiness > 61.8) ---
        # Only trade when market is trending (CHOP < 38.2) or transition (38.2-61.8)
        if chop_1w_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        # --- Price Action at Camarilla Levels with Volume Spike ---
        # Long: Price touches or goes below L3/L4 with volume spike
        touch_long = (low[i] <= l3_12h[i]) or (low[i] <= l4_12h[i])
        # Short: Price touches or goes above H3/H4 with volume spike
        touch_short = (high[i] >= h3_12h[i]) or (high[i] >= h4_12h[i])
        
        long_signal = touch_long and volume_spike[i]
        short_signal = touch_short and volume_spike[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using 12h data
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = close[entry_bar] - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if high[i] >= h3_12h[i]:  # Reached H3 level
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if low[i] <= l3_12h[i]:  # Reached L3 level
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if long_signal:
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals