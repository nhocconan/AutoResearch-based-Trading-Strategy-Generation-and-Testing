#!/usr/bin/env python3
"""
Experiment #5587: 6h Camarilla pivot levels from 1d + volume confirmation + trend filter
HYPOTHESIS: On 6h timeframe, price reactions at 1d Camarilla R3/S3 (fade) and R4/S4 (breakout) 
with volume > 1.5x average and aligned with 1w EMA50 trend capture high-probability moves. 
The 1w EMA50 provides higher timeframe trend filter, reducing false signals. Fade at R3/S3 
works in ranging markets, breakout at R4/S4 works in trending markets. ATR-based trailing 
stop (2.0x ATR) limits drawdown. Discrete position sizing (0.25) minimizes fee churn. 
Designed to work in both bull (breakouts with EMA50 support) and bear (fades at resistance) 
markets. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5587_6h_camarilla1d_1w_ema_vol_v1"
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
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate Camarilla levels from previous day's OHLC
        # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
        # Using previous day's data (shifted by 1)
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = df_1d['close'].values
        
        # Calculate levels for each day
        camarilla_r4 = c_1d + (h_1d - l_1d) * 1.1 / 2
        camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 4
        camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 4
        camarilla_s4 = c_1d - (h_1d - l_1d) * 1.1 / 2
        
        # Align to 6h timeframe (shifted by 1 day to avoid look-ahead)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    else:
        ema_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(2, 50, 14, 20)  # Camarilla needs 2 days, EMA50, ATR, volume avg
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR price reaches opposite Camarilla level (take profit)
                if price <= stop_price or price >= camarilla_s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR price reaches opposite Camarilla level (take profit)
                if price >= stop_price or price <= camarilla_r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Fade at R3/S3: price touches level and reverses
        # Breakout at R4/S4: price breaks level with conviction
        fade_long = (price <= camarilla_s3_aligned[i] * 1.001 and  # Touched or slightly above S3
                     price >= camarilla_s3_aligned[i] * 0.999 and   # Touched or slightly below S3
                     close[i] > open[i] and                         # Bullish candle
                     volume_confirmed and
                     price > ema_1w_aligned[i])                   # Uptrend filter
        
        fade_short = (price >= camarilla_r3_aligned[i] * 0.999 and  # Touched or slightly below R3
                      price <= camarilla_r3_aligned[i] * 1.001 and   # Touched or slightly above R3
                      close[i] < open[i] and                         # Bearish candle
                      volume_confirmed and
                      price < ema_1w_aligned[i])                   # Downtrend filter
        
        breakout_long = (price > camarilla_r4_aligned[i] and     # Break above R4
                         volume_confirmed and
                         price > ema_1w_aligned[i])              # Uptrend filter
        
        breakout_short = (price < camarilla_s4_aligned[i] and    # Break below S4
                          volume_confirmed and
                          price < ema_1w_aligned[i])             # Downtrend filter
        
        if fade_long or breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif fade_short or breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals