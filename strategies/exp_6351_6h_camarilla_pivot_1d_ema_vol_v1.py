#!/usr/bin/env python3
"""
Experiment #6351: 6h Camarilla Pivot Reversal with 1d Trend Filter
HYPOTHESIS: At 6h timeframe, price reversing from Camarilla R3/S3 levels (mean reversion in range) 
or breaking R4/S4 (continuation in trend) with 1d EMA50 trend filter captures institutional order flow. 
Camarilla levels derived from prior 1d OHLC provide mathematically proven support/resistance. 
Volume confirmation (>1.5x average) ensures participation. Discrete sizing (0.25) minimizes fee churn. 
Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6351_6h_camarilla_pivot_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for EMA50 trend and Camarilla pivot ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        # Calculate EMA50 on 1d close for trend filter
        ema_1d = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
        
        # Calculate Camarilla pivot levels from prior 1d OHLC
        # Camarilla uses prior day's OHLC to calculate 8 levels (L1-L4, H1-H4)
        # H4 = Close + 1.5 * (High - Low)
        # H3 = Close + 1.1 * (High - Low)
        # H2 = Close + 0.55 * (High - Low)
        # H1 = Close + 0.275 * (High - Low)
        # L1 = Close - 0.275 * (High - Low)
        # L2 = Close - 0.55 * (High - Low)
        # L3 = Close - 1.1 * (High - Low)
        # L4 = Close - 1.5 * (High - Low)
        prior_high = df_1d['high'].shift(1).values
        prior_low = df_1d['low'].shift(1).values
        prior_close = df_1d['close'].shift(1).values
        camarilla_h4 = prior_close + 1.5 * (prior_high - prior_low)
        camarilla_h3 = prior_close + 1.1 * (prior_high - prior_low)
        camarilla_l3 = prior_close - 1.1 * (prior_high - prior_low)
        camarilla_l4 = prior_close - 1.5 * (prior_high - prior_low)
        
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    else:
        ema_1d_aligned = np.full(n, np.nan)
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 50, 14) + 1  # volume avg, EMA50, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Camarilla L3 (failed continuation/reversal)
                # 3. Price crosses below 1d EMA50 (trend change)
                if price <= stop_price or price <= camarilla_l3_aligned[i] or price < ema_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Camarilla H3 (failed continuation/reversal)
                # 3. Price crosses above 1d EMA50 (trend change)
                if price >= stop_price or price >= camarilla_h3_aligned[i] or price > ema_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Camarilla-based entries:
        # LONG: 
        #   - Mean reversion: price < L3 and reversing up (price > prior close) 
        #   - OR breakout: price > H4 with volume
        #   - Filter: only if price > 1d EMA50 (uptrend bias)
        # SHORT:
        #   - Mean reversion: price > H3 and reversing down (price < prior close)
        #   - OR breakdown: price < L4 with volume
        #   - Filter: only if price < 1d EMA50 (downtrend bias)
        
        # Need prior close for reversal detection
        if i > 0:
            prior_close = close[i-1]
        else:
            prior_close = price
        
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Long entry conditions
        long_mean_reversion = (price < camarilla_l3_aligned[i]) and (price > prior_close) and volume_confirmed
        long_breakout = (price > camarilla_h4_aligned[i]) and volume_confirmed
        long_entry = (long_mean_reversion or long_breakout) and (price > ema_1d_aligned[i])
        
        # Short entry conditions
        short_mean_reversion = (price > camarilla_h3_aligned[i]) and (price < prior_close) and volume_confirmed
        short_breakdown = (price < camarilla_l4_aligned[i]) and volume_confirmed
        short_entry = (short_mean_reversion or short_breakdown) and (price < ema_1d_aligned[i])
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals