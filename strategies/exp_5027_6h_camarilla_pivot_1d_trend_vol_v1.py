#!/usr/bin/env python3
"""
Experiment #5027: 6h Camarilla Pivot Reversal + 1d Trend Filter + Volume Confirmation
HYPOTHESIS: On 6h timeframe, price reversals from Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
in alignment with 1d trend (EMA50) and volume confirmation (>1.5x average) capture high-probability swing trades. 
Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while working in both bull (breakout continuation) 
and bear (mean reversion at extremes) markets through adaptive logic based on 1d trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5027_6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: EMA50 for trend filter ===
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF EMA50 to 6h timeframe
    if len(ema_1d) > 0:
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Camarilla formula: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    if len(df_1d) >= 2:
        # Use previous day's OHLC to calculate today's pivot levels (no look-ahead)
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Shift by 1 to use previous day's data for current day's levels
        high_prev = np.concatenate([[np.nan], high_1d[:-1]])
        low_prev = np.concatenate([[np.nan], low_1d[:-1]])
        close_prev = np.concatenate([[np.nan], close_1d[:-1]])
        
        # Calculate pivot levels based on previous day
        pp = (high_prev + low_prev + close_prev) / 3.0
        range_prev = high_prev - low_prev
        
        r4 = close_prev + (range_prev * 1.1 / 2.0)
        r3 = close_prev + (range_prev * 1.1 / 4.0)
        s3 = close_prev - (range_prev * 1.1 / 4.0)
        s4 = close_prev - (range_prev * 1.1 / 2.0)
    else:
        pp = r3 = s3 = r4 = s4 = np.full(len(df_1d), np.nan)
    
    # Align HTF Camarilla levels to 6h timeframe
    if len(pp) > 0:
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pp_aligned = r3_aligned = s3_aligned = r4_aligned = s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
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
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Determine market regime based on 1d trend strength
        # Strong trend: price > 1.02 * EMA50 (bullish) or price < 0.98 * EMA50 (bearish)
        # Weak/ranging: price within 2% of EMA50
        strong_uptrend = price > ema_1d_aligned[i] * 1.02
        strong_downtrend = price < ema_1d_aligned[i] * 0.98
        ranging_market = (price >= ema_1d_aligned[i] * 0.98) and (price <= ema_1d_aligned[i] * 1.02)
        
        # Adaptive logic based on regime:
        # In strong trends: look for breakout continuation at R4/S4
        # In ranging markets: look for mean reversion at R3/S3
        
        # Long conditions
        long_breakout = strong_uptrend and (price >= r4_aligned[i]) and vol_confirm
        long_reversion = ranging_market and (price <= s3_aligned[i]) and vol_confirm
        
        # Short conditions
        short_breakout = strong_downtrend and (price <= s4_aligned[i]) and vol_confirm
        short_reversion = ranging_market and (price >= r3_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if long_breakout or long_reversion:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_breakout or short_reversion:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals