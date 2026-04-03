#!/usr/bin/env python3
"""
Experiment #074: 1h Camarilla Pivot + 4h/1d Volume + Session Filter
HYPOTHESIS: 1h Camarilla pivot reversals aligned with 4h/1d volume spikes during active UTC 08-20 session capture mean-reversion opportunities in both bull and bear markets. Using HTF (4h/1d) for volume confirmation reduces false signals while 1h provides precise entry timing. Target: 60-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_074_1h_camarilla_4h_1d_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for volume MA (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    vol_ma_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # === HTF: 1d data for volume MA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 1h Indicators: Camarilla Pivot (using previous bar) ===
    # Camarilla levels based on previous bar's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    range_val = prev_high - prev_low
    camarilla_h3 = prev_close + range_val * 1.1 / 4
    camarilla_l3 = prev_close - range_val * 1.1 / 4
    camarilla_h4 = prev_close + range_val * 1.1 / 2
    camarilla_l4 = prev_close - range_val * 1.1 / 2
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Session Filter: UTC 08-20 (active trading hours) ===
    # open_time is already datetime64[ms], access hour via index
    hours = prices.index.hour  # Pre-compute once before loop
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # Warmup for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(camarilla_h4[i]) or
            np.isnan(camarilla_l4[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_4h_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 08-20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h = volume[i]
        vol_1d = volume[i]
        vol_spike_4h = vol_4h > 2.0 * vol_ma_4h_aligned[i]
        vol_spike_1d = vol_1d > 1.5 * vol_ma_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Exit at opposite Camarilla level (H4/L4) with volume confirmation
            if position_side > 0:  # Long
                if high[i] >= camarilla_h4[i] and vol_spike_4h:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if low[i] <= camarilla_l4[i] and vol_spike_4h:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price crosses above L3 with volume spike
        # Short: Price crosses below H3 with volume spike
        if vol_spike_4h and vol_spike_1d:
            # Long entry
            if low[i] <= camarilla_l3[i] and close[i] > camarilla_l3[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry
            elif high[i] >= camarilla_h3[i] and close[i] < camarilla_h3[i]:
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