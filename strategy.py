#!/usr/bin/env python3
"""
Experiment #159: 6h Camarilla Pivot + 12h Trend Filter + Volume Spike

HYPOTHESIS: Camarilla pivot levels derived from 1d OHLC provide institutional support/resistance 
on 6h timeframe. Price approaching S3/R3 levels with volume confirmation offers high-probability 
mean reversion entries, while breaks of S4/R4 with 12h trend alignment capture continuation moves. 
The 12h EMA50 filter ensures we trade with the higher timeframe trend, reducing whipsaw in 
choppy markets. Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to 
minimize fee drag while capturing high-probability pivot reactions and breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(50) on 12h close
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    else:
        ema_50_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d OHLC
    camarilla_s1 = np.full(n, np.nan)
    camarilla_s2 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_r2 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_close = np.full(n, np.nan)
    
    if len(df_1d) >= 1:
        # Vectorized calculation for all 1d bars
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Camarilla formulas
        camarilla_pp = (high_1d + low_1d + close_1d) / 3
        camarilla_range = high_1d - low_1d
        
        camarilla_s1_vec = camarilla_pp - camarilla_range * 1.0 / 12
        camarilla_s2_vec = camarilla_pp - camarilla_range * 2.0 / 12
        camarilla_s3_vec = camarilla_pp - camarilla_range * 3.0 / 12
        camarilla_s4_vec = camarilla_pp - camarilla_range * 4.0 / 12
        camarilla_r1_vec = camarilla_pp + camarilla_range * 1.0 / 12
        camarilla_r2_vec = camarilla_pp + camarilla_range * 2.0 / 12
        camarilla_r3_vec = camarilla_pp + camarilla_range * 3.0 / 12
        camarilla_r4_vec = camarilla_pp + camarilla_range * 4.0 / 12
        
        # Align to 6h timeframe
        camarilla_s1 = align_htf_to_ltf(prices, df_1d, camarilla_s1_vec)
        camarilla_s2 = align_htf_to_ltf(prices, df_1d, camarilla_s2_vec)
        camarilla_s3 = align_htf_to_ltf(prices, df_1d, camarilla_s3_vec)
        camarilla_s4 = align_htf_to_ltf(prices, df_1d, camarilla_s4_vec)
        camarilla_r1 = align_htf_to_ltf(prices, df_1d, camarilla_r1_vec)
        camarilla_r2 = align_htf_to_ltf(prices, df_1d, camarilla_r2_vec)
        camarilla_r3 = align_htf_to_ltf(prices, df_1d, camarilla_r3_vec)
        camarilla_r4 = align_htf_to_ltf(prices, df_1d, camarilla_r4_vec)
        camarilla_close = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF EMA50 and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s4[i]) or 
            np.isnan(camarilla_r4[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: 12h EMA50 direction ---
        price_above_12h_ema = close[i] > ema_50_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_50_12h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Camarilla Pivot Conditions ---
        # Mean reversion at S3/R3 (touch and bounce)
        near_s3 = low[i] <= camarilla_s3[i] * 1.002 and high[i] >= camarilla_s3[i] * 0.998
        near_r3 = high[i] >= camarilla_r3[i] * 0.998 and low[i] <= camarilla_r3[i] * 1.002
        
        # Breakout continuation at S4/R4 (close beyond level)
        breakout_s4 = close[i] < camarilla_s4[i]
        breakout_r4 = close[i] > camarilla_r4[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite pivot level
                if position_side == 1 and high[i] >= camarilla_r3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite pivot level
                if position_side == -1 and low[i] <= camarilla_s3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long mean reversion: touch S3 + volume spike + price above 12h EMA50
        long_mr = near_s3 and volume_spike and price_above_12h_ema
        
        # Short mean reversion: touch R3 + volume spike + price below 12h EMA50
        short_mr = near_r3 and volume_spike and price_below_12h_ema
        
        # Long breakout: close below S4 + volume spike + price above 12h EMA50
        long_break = breakout_s4 and volume_spike and price_above_12h_ema
        
        # Short breakout: close above R4 + volume spike + price below 12h EMA50
        short_break = breakout_r4 and volume_spike and price_below_12h_ema
        
        if long_mr or long_break:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_mr or short_break:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals