#!/usr/bin/env python3
"""
Experiment #5251: 6h Camarilla Pivot + Volume Spike + Regime Filter
HYPOTHESIS: On 6h timeframe, Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from 1d timeframe provide institutional support/resistance. Enter long at S3 with volume spike (>2.0x) in bullish regime (price > 1d EMA50), enter short at R3 with volume spike in bearish regime (price < 1d EMA50). Breakouts at R4/S4 with volume spike continue the trend. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Works in bull markets (mean reversion at S3/R3, breakout at R4) and bear markets (mean reversion at R3/S3, breakout at S4). Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5251_6h_camarilla_pivot_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Camarilla pivot and regime
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Camarilla Pivot Levels (using prior day's OHLC) ===
    if len(df_1d) >= 1:
        # Prior day OHLC (shifted by 1 to avoid look-ahead)
        prior_high = df_1d['high'].shift(1).values
        prior_low = df_1d['low'].shift(1).values
        prior_close = df_1d['close'].shift(1).values
        
        # Camarilla calculations
        pivot = (prior_high + prior_low + prior_close) / 3.0
        range_hl = prior_high - prior_low
        
        # Resistance levels
        r3 = pivot + range_hl * 1.1 / 2.0
        r4 = pivot + range_hl * 1.1
        # Support levels
        s3 = pivot - range_hl * 1.1 / 2.0
        s4 = pivot - range_hl * 1.1
        
        # Align to 6h timeframe (shift(1) in align_htf_to_ltf ensures prior day only)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
        
        # 1d EMA50 for regime filter
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        r3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
        ema_50_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation (2.0x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Regime filter: bullish if price > 1d EMA50, bearish if price < 1d EMA50
        regime_bullish = price > ema_50_aligned[i]
        regime_bearish = price < ema_50_aligned[i]
        
        # Mean reversion at S3/R3 (long at S3 in bullish regime, short at R3 in bearish regime)
        mean_revert_long = (price <= s3_aligned[i]) and regime_bullish and vol_confirm
        mean_revert_short = (price >= r3_aligned[i]) and regime_bearish and vol_confirm
        
        # Breakout continuation at R4/S4 (breakout with volume in regime direction)
        breakout_long = (price >= r4_aligned[i]) and regime_bullish and vol_confirm
        breakout_short = (price <= s4_aligned[i]) and regime_bearish and vol_confirm
        
        # Final entry conditions
        if mean_revert_long or breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif mean_revert_short or breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals