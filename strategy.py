#!/usr/bin/env python3
"""
Experiment #4711: 6h Camarilla Pivot Reversal + 1d Trend Filter
HYPOTHESIS: At 6h timeframe, price reactions to 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with 1d EMA50 trend filter captures institutional order flow. In bull/bear markets, price often reverses at R3/S3 (80% probability) or breaks through R4/S4 with continuation. This strategy targets 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in ranging markets (reversions at R3/S3) and trending markets (breakouts at R4/S4).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4711_6h_camarilla_pivot_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Camarilla pivots and EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Camarilla pivot levels from prior day ===
    if len(df_1d) >= 2:
        # Use prior day's OHLC (shifted by 1 to avoid look-ahead)
        ph = np.concatenate([[np.nan], df_1d['high'].values[:-1]])   # prior day high
        pl = np.concatenate([[np.nan], df_1d['low'].values[:-1]])    # prior day low
        pc = np.concatenate([[np.nan], df_1d['close'].values[:-1]])  # prior day close
        
        # Camarilla calculations
        pivot = (ph + pl + pc) / 3
        range_ = ph - pl
        
        # Resistance levels
        r3 = pivot + range_ * 1.1 / 2
        r4 = pivot + range_ * 1.1
        # Support levels
        s3 = pivot - range_ * 1.1 / 2
        s4 = pivot - range_ * 1.1
        
        # Align to 6h timeframe
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: EMA50 for trend filter ===
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF EMA50 to 6h timeframe
    if len(ema_1d) > 0:
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation ===
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
    
    warmup = max(2, 20, 14, 50)  # Prior day, Volume MA, ATR, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.3x)
        vol_confirm = vol_ratio[i] > 1.3
        
        # Mean reversion at R3/S3 (80% probability zone)
        mean_revert_long = (price <= s3_aligned[i] * 1.001) and vol_confirm  # slight buffer
        mean_revert_short = (price >= r3_aligned[i] * 0.999) and vol_confirm  # slight buffer
        
        # Breakout continuation at R4/S4 with trend alignment
        breakout_long = (price >= r4_aligned[i] * 0.999) and (price > ema_1d_aligned[i]) and vol_confirm
        breakout_short = (price <= s4_aligned[i] * 1.001) and (price < ema_1d_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if mean_revert_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif mean_revert_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        elif breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals