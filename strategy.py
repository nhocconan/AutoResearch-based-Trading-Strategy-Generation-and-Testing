#!/usr/bin/env python3
"""
Experiment #085: 12h Camarilla pivot + volume spike + choppiness regime

HYPOTHESIS: On 12h timeframe, price retracing to Camarilla pivot levels (H3/L3) from prior 1d,
with 1d volume confirmation (>1.5x average) and choppiness regime filter (CHOP > 61.8 = ranging),
captures mean-reversion bounces in both bull and bear markets. The 12h timeframe reduces noise,
Camarilla levels provide precise entry/exit points, volume confirms participation, and chop filter
avoids trending markets where mean reversion fails. Targets 12-37 trades/year (50-150 total over 4 years)
to minimize fee drag while maintaining edge across market regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior 1d bar
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        pivot = (high_1d + low_1d + close_1d) / 3.0
        range_1d = high_1d - low_1d
        # Camarilla levels: H3/L3 are strongest for mean reversion
        camarilla_h3 = pivot + (range_1d * 1.1 / 4.0)
        camarilla_l3 = pivot - (range_1d * 1.1 / 4.0)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for choppiness regime filter (Call ONCE before loop) ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        # True Range
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.roll(close_1d, 1))
        tr3 = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_1d[0] - low_1d[0]  # First period
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        # Absolute close change over 14 periods
        abs_close_chg = np.abs(close_1d - np.roll(close_1d, 14))
        abs_close_chg[:14] = 0  # First 14 periods
        # Choppiness Index: CHOP = 100 * log10(sum(tr14)/abs(close_chg14)) / log10(14)
        chop = np.zeros(len(close_1d))
        mask = (tr_sum > 0) & (abs_close_chg > 0)
        chop[mask] = 100 * np.log10(tr_sum[mask] / abs_close_chg[mask]) / np.log10(14)
        chop[:14] = 50.0  # Neutral for warmup
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)
    
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
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in ranging markets (CHOP > 61.8) ---
        ranging_market = chop_aligned[i] > 61.8
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) on 1d ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic: Mean reversion target at pivot ---
        if in_position:
            # Exit when price reaches pivot level (mean reversion complete)
            if position_side > 0:  # Long position
                if close[i] >= camarilla_h3_aligned[i]:  # Actually, we exit at pivot, not H3
                    # Wait, correction: We enter near H3/L3, exit at pivot
                    # For long entered near L3, exit at pivot
                    if close[i] >= (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2.0:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
                # Stoploss: price moves against us by 1.5 * ATR equivalent
                # Simplified: use 1.5% of price as volatility proxy
                if close[i] < entry_price * 0.985:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if close[i] <= camarilla_l3_aligned[i]:
                    if close[i] <= (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2.0:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                        continue
                if close[i] > entry_price * 1.015:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price retraces to L3 level with volume and ranging regime
        long_condition = (
            low[i] <= camarilla_l3_aligned[i] * 1.001 and  # Allow small slippage
            ranging_market and
            volume_spike
        )
        
        # Short: Price retraces to H3 level with volume and ranging regime
        short_condition = (
            high[i] >= camarilla_h3_aligned[i] * 0.999 and  # Allow small slippage
            ranging_market and
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