#!/usr/bin/env python3
"""
Experiment #107: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts in direction of 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) with volume confirmation (>1.8x) capture institutional order flow. Uses discrete sizing (0.25) and ATR stoploss (2.0*ATR). Target: 75-150 total trades over 4 years (19-37/year). Works in bull/bear via pivot-based directional filter and volatility stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_107_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla pivot levels for daily timeframe
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # Pivot point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels:
    # R4 = PP + Range * 1.1/2
    # R3 = PP + Range * 1.1/4
    # S3 = PP - Range * 1.1/4
    # S4 = PP - Range * 1.1/2
    r4_1d = pp_1d + range_1d * 1.1 / 2.0
    r3_1d = pp_1d + range_1d * 1.1 / 4.0
    s3_1d = pp_1d - range_1d * 1.1 / 4.0
    s4_1d = pp_1d - range_1d * 1.1 / 2.0
    
    # Determine pivot-based signal:
    # If price > R4: strong bullish breakout (signal = 1)
    # If price > R3: bullish bias (signal = 0.5)
    # If price < S3: bearish bias (signal = -0.5)
    # If price < S4: strong bearish breakout (signal = -1)
    # Else: neutral (signal = 0)
    camarilla_signal = np.zeros(len(close_1d))
    camarilla_signal[close_1d > r4_1d] = 1.0
    camarilla_signal[(close_1d > r3_1d) & (close_1d <= r4_1d)] = 0.5
    camarilla_signal[(close_1d < s3_1d) & (close_1d >= s4_1d)] = -0.5
    camarilla_signal[close_1d < s4_1d] = -1.0
    
    camarilla_signal_aligned = align_htf_to_ltf(prices, df_1d, camarilla_signal)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
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
            np.isnan(vol_ratio[i]) or np.isnan(camarilla_signal_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Daily Camarilla Pivot Signal ---
        camarilla_bullish = camarilla_signal_aligned[i] > 0
        camarilla_bearish = camarilla_signal_aligned[i] < 0
        camarilla_strong_bullish = camarilla_signal_aligned[i] >= 0.5
        camarilla_strong_bearish = camarilla_signal_aligned[i] <= -0.5
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~36h on 6h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND bullish Camarilla bias
            if breakout_up and camarilla_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish Camarilla bias
            elif breakout_down and camarilla_bearish:
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