#!/usr/bin/env python3
"""
Experiment #279: 6h Donchian(20) Breakout + 12h Volume Spike + 1d Regime Filter

HYPOTHESIS: Donchian channel breakouts on 6h timeframe capture intermediate-term trends, 
filtered by 12h volume spikes to confirm institutional participation and 1d choppiness 
regime to avoid false signals in ranging markets. This combination should work in both 
bull and bear markets by only taking breakouts with volume confirmation in trending regimes.
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        # Volume ratio: current volume / 20-period average volume
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup period
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for choppiness regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
        chop_1d = np.full(len(close_1d), np.nan)
        valid = (sum_tr_14 > 0) & (max_high_14 > min_low_14) & ~(np.isnan(sum_tr_14) | np.isnan(max_high_14) | np.isnan(min_low_14))
        chop_1d[valid] = 100 * np.log10(sum_tr_14[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
        
        # Align to 6h timeframe
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, 50.0)  # Neutral chop value
    
    # === 6h Indicators: Donchian Channel(20) ===
    # Calculate Donchian channels on 6h data
    max_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    min_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(max_high_20[i]) or np.isnan(min_low_20[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (Choppiness < 61.8) ---
        # Avoid strong ranging regimes where breakouts fail
        if chop_1d_aligned[i] >= 61.8:
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (ratio > 1.5) ---
        if vol_ratio_12h_aligned[i] < 1.5:
            signals[i] = 0.0
            continue
        
        # --- Donchian Breakout Signals ---
        # Long breakout: price closes above 20-period high
        long_breakout = close[i] > max_high_20[i-1]  # Use previous bar's high to avoid look-ahead
        # Short breakout: price closes below 20-period low
        short_breakout = close[i] < min_low_20[i-1]  # Use previous bar's low to avoid look-ahead
        
        # --- Exit Logic (ATR-based stoploss and time-based exit) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                # Exit on stoploss hit or Donchian reversal
                if low[i] < stop_level or close[i] < min_low_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                # Exit on stoploss hit or Donchian reversal
                if high[i] > stop_level or close[i] > max_high_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above upper band with volume confirmation
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakout below lower band with volume confirmation
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #279: 6h Donchian(20) Breakout + 12h Volume Spike + 1d Regime Filter

HYPOTHESIS: Donchian channel breakouts on 6h timeframe capture intermediate-term trends, 
filtered by 12h volume spikes to confirm institutional participation and 1d choppiness 
regime to avoid false signals in ranging markets. This combination should work in both 
bull and bear markets by only taking breakouts with volume confirmation in trending regimes.
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        # Volume ratio: current volume / 20-period average volume
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup period
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for choppiness regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        # Sum of TR over 14 periods
        sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
        chop_1d = np.full(len(close_1d), np.nan)
        valid = (sum_tr_14 > 0) & (max_high_14 > min_low_14) & ~(np.isnan(sum_tr_14) | np.isnan(max_high_14) | np.isnan(min_low_14))
        chop_1d[valid] = 100 * np.log10(sum_tr_14[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
        
        # Align to 6h timeframe
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, 50.0)  # Neutral chop value
    
    # === 6h Indicators: Donchian Channel(20) ===
    # Calculate Donchian channels on 6h data
    max_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    min_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(max_high_20[i]) or np.isnan(min_low_20[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (Choppiness < 61.8) ---
        # Avoid strong ranging regimes where breakouts fail
        if chop_1d_aligned[i] >= 61.8:
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (ratio > 1.5) ---
        if vol_ratio_12h_aligned[i] < 1.5:
            signals[i] = 0.0
            continue
        
        # --- Donchian Breakout Signals ---
        # Long breakout: price closes above 20-period high
        long_breakout = close[i] > max_high_20[i-1]  # Use previous bar's high to avoid look-ahead
        # Short breakout: price closes below 20-period low
        short_breakout = close[i] < min_low_20[i-1]  # Use previous bar's low to avoid look-ahead
        
        # --- Exit Logic (ATR-based stoploss and time-based exit) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                # Exit on stoploss hit or Donchian reversal
                if low[i] < stop_level or close[i] < min_low_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                # Exit on stoploss hit or Donchian reversal
                if high[i] > stop_level or close[i] > max_high_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above upper band with volume confirmation
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakout below lower band with volume confirmation
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals