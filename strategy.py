#!/usr/bin/env python3
"""
Experiment #047: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: Price breaking 6h Donchian(20) channels with weekly pivot trend alignment (price above/below weekly pivot) and volume spike (>2.0x) captures strong momentum while minimizing false breakouts. Weekly pivot provides robust trend filter that works in both bull (breakouts above pivot) and bear (breakdowns below pivot) markets. Discrete sizing (0.25) and ATR(14) stoploss (2.5) manage risk. Target: 75-200 total trades over 4 years (19-50/year) for statistical validity and low fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_047_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d and 1w data for pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    wk_pivot = (wk_high + wk_low + wk_close) / 3.0
    wk_range = wk_high - wk_low
    wk_r1 = 2 * wk_pivot - wk_low
    wk_s1 = 2 * wk_pivot - wk_high
    wk_r2 = wk_pivot + wk_range
    wk_s2 = wk_pivot - wk_range
    wk_r3 = wk_high + 2 * (wk_pivot - wk_low)
    wk_s3 = wk_low - 2 * (wk_high - wk_pivot)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed week only)
    wk_pivot_aligned = align_htf_to_ltf(prices, df_1w, wk_pivot)
    wk_r3_aligned = align_htf_to_ltf(prices, df_1w, wk_r3)
    wk_s3_aligned = align_htf_to_ltf(prices, df_1w, wk_s3)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(wk_pivot_aligned[i]) or
            np.isnan(wk_r3_aligned[i]) or np.isnan(wk_s3_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Trend Filter: Weekly pivot alignment ---
        # Uptrend: price above weekly R3 (strong bullish)
        # Downtrend: price below weekly S3 (strong bearish)
        uptrend = price > wk_r3_aligned[i]
        downtrend = price < wk_s3_aligned[i]
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 20 bars (~5d on 6h) to avoid overtrading
            if bars_since_entry > 20:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND strong uptrend (above weekly R3)
            if breakout_up and uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND strong downtrend (below weekly S3)
            elif breakout_down and downtrend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #047: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: Price breaking 6h Donchian(20) channels with weekly pivot trend alignment (price above/below weekly pivot) and volume spike (>2.0x) captures strong momentum while minimizing false breakouts. Weekly pivot provides robust trend filter that works in both bull (breakouts above pivot) and bear (breakdowns below pivot) markets. Discrete sizing (0.25) and ATR(14) stoploss (2.5) manage risk. Target: 75-200 total trades over 4 years (19-50/year) for statistical validity and low fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_047_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d and 1w data for pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    wk_pivot = (wk_high + wk_low + wk_close) / 3.0
    wk_range = wk_high - wk_low
    wk_r1 = 2 * wk_pivot - wk_low
    wk_s1 = 2 * wk_pivot - wk_high
    wk_r2 = wk_pivot + wk_range
    wk_s2 = wk_pivot - wk_range
    wk_r3 = wk_high + 2 * (wk_pivot - wk_low)
    wk_s3 = wk_low - 2 * (wk_high - wk_pivot)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed week only)
    wk_pivot_aligned = align_htf_to_ltf(prices, df_1w, wk_pivot)
    wk_r3_aligned = align_htf_to_ltf(prices, df_1w, wk_r3)
    wk_s3_aligned = align_htf_to_ltf(prices, df_1w, wk_s3)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(wk_pivot_aligned[i]) or
            np.isnan(wk_r3_aligned[i]) or np.isnan(wk_s3_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Trend Filter: Weekly pivot alignment ---
        # Uptrend: price above weekly R3 (strong bullish)
        # Downtrend: price below weekly S3 (strong bearish)
        uptrend = price > wk_r3_aligned[i]
        downtrend = price < wk_s3_aligned[i]
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 20 bars (~5d on 6h) to avoid overtrading
            if bars_since_entry > 20:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND strong uptrend (above weekly R3)
            if breakout_up and uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND strong downtrend (below weekly S3)
            elif breakout_down and downtrend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

}