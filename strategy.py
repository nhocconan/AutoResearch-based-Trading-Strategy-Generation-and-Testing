#!/usr/bin/env python3
"""
Experiment #335: 6h Elder Ray + Weekly Pivot + Volume Confirmation

HYPOTHESIS: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13.
Combined with 1w pivot direction for trend filter and 1d volume confirmation, captures
institutional participation in 6h trends. Works in both bull/bear by adapting to regime
via pivot levels. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 5:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        pivot = (high_1w + low_1w + close_1w) / 3.0
        r1 = 2 * pivot - low_1w
        s1 = 2 * pivot - high_1w
        r2 = pivot + (high_1w - low_1w)
        s2 = pivot - (high_1w - low_1w)
        r3 = high_1w + 2 * (pivot - low_1w)
        s3 = low_1w - 2 * (high_1w - pivot)
        
        # Align to LTF (6h) with shift(1) for completed bars only
        pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1w, r3 + (r3 - s3))  # R4 = R3 + (R3-S3)
        s4_aligned = align_htf_to_ltf(prices, df_1w, s3 - (r3 - s3))  # S4 = S3 - (R3-S3)
    else:
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter from Weekly Pivot ---
        # Uptrend: price above weekly pivot
        # Downtrend: price below weekly pivot
        uptrend = close[i] > pivot_aligned[i]
        downtrend = close[i] < pivot_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.3x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.3
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly R3 (resistance)
                if close[i] >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly S3 (support)
                if close[i] <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Bull Power > 0 (buying pressure) in uptrend with volume confirmation
        long_condition = (
            bull_power[i] > 0 and 
            uptrend and 
            volume_spike
        )
        
        # Short: Bear Power < 0 (selling pressure) in downtrend with volume confirmation
        short_condition = (
            bear_power[i] < 0 and 
            downtrend and 
            volume_spike
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #335: 6h Elder Ray + Weekly Pivot + Volume Confirmation

HYPOTHESIS: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA13.
Combined with 1w pivot direction for trend filter and 1d volume confirmation, captures
institutional participation in 6h trends. Works in both bull/bear by adapting to regime
via pivot levels. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 5:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        pivot = (high_1w + low_1w + close_1w) / 3.0
        r1 = 2 * pivot - low_1w
        s1 = 2 * pivot - high_1w
        r2 = pivot + (high_1w - low_1w)
        s2 = pivot - (high_1w - low_1w)
        r3 = high_1w + 2 * (pivot - low_1w)
        s3 = low_1w - 2 * (high_1w - pivot)
        
        # Align to LTF (6h) with shift(1) for completed bars only
        pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1w, r3 + (r3 - s3))  # R4 = R3 + (R3-S3)
        s4_aligned = align_htf_to_ltf(prices, df_1w, s3 - (r3 - s3))  # S4 = S3 - (R3-S3)
    else:
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter from Weekly Pivot ---
        # Uptrend: price above weekly pivot
        # Downtrend: price below weekly pivot
        uptrend = close[i] > pivot_aligned[i]
        downtrend = close[i] < pivot_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.3x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.3
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly R3 (resistance)
                if close[i] >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly S3 (support)
                if close[i] <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Bull Power > 0 (buying pressure) in uptrend with volume confirmation
        long_condition = (
            bull_power[i] > 0 and 
            uptrend and 
            volume_spike
        )
        
        # Short: Bear Power < 0 (selling pressure) in downtrend with volume confirmation
        short_condition = (
            bear_power[i] < 0 and 
            downtrend and 
            volume_spike
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals