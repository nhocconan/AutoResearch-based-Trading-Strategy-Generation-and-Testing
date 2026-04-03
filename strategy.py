#!/usr/bin/env python3
"""
Experiment #443: 4h Donchian(20) Breakout + 12h Volume Spike + 1d HMA Trend Filter

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by 12h volume spikes (>2x average) 
and aligned with 1d HMA(21) trend, capture high-probability momentum moves in both bull and bear markets. 
The Donchian structure provides objective breakout levels, volume confirms institutional participation, 
and the HMA trend filter ensures we only trade with the higher timeframe direction. 
ATR-based stoploss manages risk. Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years) 
to minimize fee drag while capturing strong trending moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_12h_vol_1d_hma_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = np.full_like(close_1d, np.nan)
        wma_full = np.full_like(close_1d, np.nan)
        
        if len(close_1d) >= half_len:
            wma_half[half_len-1:] = wma(close_1d, half_len)
        if len(close_1d) >= 21:
            wma_full[20:] = wma(close_1d, 21)
        
        # 2*WMA(half) - WMA(full)
        diff = 2 * wma_half - wma_full
        hma_21 = np.full_like(close_1d, np.nan)
        
        if len(diff) >= sqrt_len:
            wma_diff = wma(diff[~np.isnan(diff)], sqrt_len) if np.sum(~np.isnan(diff)) >= sqrt_len else np.array([])
            if len(wma_diff) > 0:
                hma_21[20 + half_len - 1:] = wma_diff
        
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high[19:] = high_series.rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = low_series.rolling(window=20, min_periods=20).min().values[19:]
    
    # ATR(14) for stoploss
    atr_14 = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for j in range(1, n):
            tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(hma_21_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based trailing stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                # Update highest high since entry
                highest_since_entry = max(highest_since_entry, high[i])
                stop_level = highest_since_entry - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks below Donchian low (trend reversal)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Update lowest low since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_level = lowest_since_entry + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks above Donchian high (trend reversal)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume confirmation and uptrend (price > HMA)
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above upper band
            vol_ratio_12h_aligned[i] > 2.0 and  # Volume spike (>2x average)
            close[i] > hma_21_aligned[i]  # Uptrend filter
        )
        
        # Short: Price breaks below Donchian low with volume confirmation and downtrend (price < HMA)
        short_condition = (
            close[i] < donchian_low[i] and  # Breakdown below lower band
            vol_ratio_12h_aligned[i] > 2.0 and  # Volume spike (>2x average)
            close[i] < hma_21_aligned[i]  # Downtrend filter
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #443: 4h Donchian(20) Breakout + 12h Volume Spike + 1d HMA Trend Filter

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by 12h volume spikes (>2x average) 
and aligned with 1d HMA(21) trend, capture high-probability momentum moves in both bull and bear markets. 
The Donchian structure provides objective breakout levels, volume confirms institutional participation, 
and the HMA trend filter ensures we only trade with the higher timeframe direction. 
ATR-based stoploss manages risk. Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years) 
to minimize fee drag while capturing strong trending moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_12h_vol_1d_hma_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = np.full_like(close_1d, np.nan)
        wma_full = np.full_like(close_1d, np.nan)
        
        if len(close_1d) >= half_len:
            wma_half[half_len-1:] = wma(close_1d, half_len)
        if len(close_1d) >= 21:
            wma_full[20:] = wma(close_1d, 21)
        
        # 2*WMA(half) - WMA(full)
        diff = 2 * wma_half - wma_full
        hma_21 = np.full_like(close_1d, np.nan)
        
        if len(diff) >= sqrt_len:
            wma_diff = wma(diff[~np.isnan(diff)], sqrt_len) if np.sum(~np.isnan(diff)) >= sqrt_len else np.array([])
            if len(wma_diff) > 0:
                hma_21[20 + half_len - 1:] = wma_diff
        
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high[19:] = high_series.rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = low_series.rolling(window=20, min_periods=20).min().values[19:]
    
    # ATR(14) for stoploss
    atr_14 = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for j in range(1, n):
            tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(hma_21_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based trailing stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                # Update highest high since entry
                highest_since_entry = max(highest_since_entry, high[i])
                stop_level = highest_since_entry - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks below Donchian low (trend reversal)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Update lowest low since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_level = lowest_since_entry + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks above Donchian high (trend reversal)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume confirmation and uptrend (price > HMA)
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above upper band
            vol_ratio_12h_aligned[i] > 2.0 and  # Volume spike (>2x average)
            close[i] > hma_21_aligned[i]  # Uptrend filter
        )
        
        # Short: Price breaks below Donchian low with volume confirmation and downtrend (price < HMA)
        short_condition = (
            close[i] < donchian_low[i] and  # Breakdown below lower band
            vol_ratio_12h_aligned[i] > 2.0 and  # Volume spike (>2x average)
            close[i] < hma_21_aligned[i]  # Downtrend filter
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals