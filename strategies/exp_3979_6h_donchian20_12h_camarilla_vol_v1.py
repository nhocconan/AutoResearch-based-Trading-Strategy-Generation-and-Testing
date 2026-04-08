#!/usr/bin/env python3
"""
Experiment #3979: 6h Donchian(20) breakout + 12h Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Camarilla pivot levels (breakout at R4/S4 for continuation, fade at R3/S3) capture multi-day swings with controlled frequency. Volume > 2.0x MA(30) confirms breakout strength. ATR(20) trailing stop (2.0x) manages risk. Discrete sizing (0.25) reduces fee drag. Target: 100-200 trades over 4 years (25-50/year). Works in bull/bear via pivot structure as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3979_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels ===
    df_12h = get_htf_data(prices, '12h')
    # Typical price for pivot calculation
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    # Previous day's (12h bar's) high, low, close for Camarilla
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    # Camarilla levels: R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low)
    # S3 = Close - 1.1*(High-Low), S4 = Close - 1.5*(High-Low)
    camarilla_r4 = close_12h + 1.5 * (high_12h - low_12h)
    camarilla_r3 = close_12h + 1.1 * (high_12h - low_12h)
    camarilla_s3 = close_12h - 1.1 * (high_12h - low_12h)
    camarilla_s4 = close_12h - 1.5 * (high_12h - low_12h)
    # Align to 6h timeframe (shift by 1 for completed 12h bar)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(30) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[30:] = volume[30:] / vol_ma[30:]
    
    # === 6h Indicators: ATR(20) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 30, 20)  # DC lookback, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Camarilla S3 (mean reversion level)
                elif price < s3_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Camarilla R3 (mean reversion level)
                elif price > r3_12h_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) to filter noise
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Determine Camarilla zone: breakout at R4/S4, fade at R3/S3
            # Long logic: breakout above R4 OR fade from S3 with bullish bias
            long_breakout = price > r4_12h_aligned[i-1]
            long_fade = price < s3_12h_aligned[i-1] and price > s4_12h_aligned[i-1]  # Oversold bounce
            # Short logic: breakdown below S4 OR fade from R3 with bearish bias
            short_breakout = price < s4_12h_aligned[i-1]
            short_fade = price > r3_12h_aligned[i-1] and price < r4_12h_aligned[i-1]  # Overbought rejection
            
            if (long_breakout or long_fade) and not (short_breakout or short_fade):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif (short_breakout or short_fade) and not (long_breakout or long_fade):
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals