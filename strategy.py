#!/usr/bin/env python3
"""
Experiment #3775: 6h Donchian(20) breakout + 1w Camarilla pivot levels + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture intermediate swings, with 1w Camarilla pivot levels identifying key institutional support/resistance. Breakouts above R4 or below S4 indicate strong momentum continuation, while reversals at R3/S3 provide mean-reversion opportunities. Volume confirmation (>1.5x average) ensures participation. Works in bull markets (breakouts above R4) and bear markets (breakdowns below S4). Position size 0.25 manages drawdown. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3775_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels
    # Camarilla: R4 = close + ((high - low) * 1.1/2), R3 = close + ((high - low) * 1.1/4)
    #            S3 = close - ((high - low) * 1.1/4), S4 = close - ((high - low) * 1.1/2)
    camarilla_r4 = np.full(len(close_1w), np.nan)
    camarilla_r3 = np.full(len(close_1w), np.nan)
    camarilla_s3 = np.full(len(close_1w), np.nan)
    camarilla_s4 = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if i < 1:  # Need at least 1 week of data
            continue
        hl_range = high_1w[i] - low_1w[i]
        camarilla_r4[i] = close_1w[i] + (hl_range * 1.1 / 2)
        camarilla_r3[i] = close_1w[i] + (hl_range * 1.1 / 4)
        camarilla_s3[i] = close_1w[i] - (hl_range * 1.1 / 4)
        camarilla_s4[i] = close_1w[i] - (hl_range * 1.1 / 2)
    
    # Align 1w Camarilla levels to 6h timeframe (shifted by 1 for completed 1w bar)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                # Using fixed ATR approximation for 6h: 0.02 * price (2%)
                atr_approx = 0.02 * price
                if price < highest_since_entry - 2.0 * atr_approx:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                atr_approx = 0.02 * price
                if price > lowest_since_entry + 2.0 * atr_approx:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above 1w R4 (strong breakout)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > r4_1w_aligned[i]):     # Above 1w Camarilla R4
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below 1w S4 (strong breakdown)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < s4_1w_aligned[i]):     # Below 1w Camarilla S4
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            # Long reversal: Price breaks below Donchian lower band but holds above 1w S3 (mean reversion)
            elif (price < lowest_low[i-1] and    # Break below Donchian low
                  price > s3_1w_aligned[i]):     # But above 1w Camarilla S3 (support)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short reversal: Price breaks above Donchian upper band but holds below 1w R3 (mean reversion)
            elif (price > highest_high[i-1] and  # Break above Donchian high
                  price < r3_1w_aligned[i]):     # But below 1w Camarilla R3 (resistance)
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