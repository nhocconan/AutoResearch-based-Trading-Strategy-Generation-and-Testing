#!/usr/bin/env python3
"""
Experiment #295: 6h Camarilla Pivot Levels with Weekly Trend Filter and Volume Confirmation

HYPOTHESIS: Camarilla pivot levels derived from daily bars act as intraday support/resistance.
In strong weekly trends (price above/below weekly EMA20), breaks of R4/S4 levels with volume
confirmation (>1.8x average volume) signal continuation. Fades at R3/S3 levels with volume
provide mean reversion opportunities in ranging markets. 6h timeframe balances signal quality
and trade frequency (target: 12-37 trades/year). Works in bull markets (continuation breaks),
bear markets (failed continuations reverse sharply), and ranging markets (mean reversion at
extremes). ATR-based stoploss manages risk.

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.25) to minimize churn
- Volume confirmation threshold: 1.8x average volume
- Minimum holding period: 2 bars to reduce churn
- Warmup period: 100 bars for stable HTF alignment
- Long: Break above R4 with volume + weekly uptrend OR fade at R3 with volume + weekly downtrend
- Short: Break below S4 with volume + weekly downtrend OR fade at S3 with volume + weekly uptrend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_295_6h_camarilla_weekly_vol_v1"
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
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        diff = h - l
        camarilla_h4[i] = c + diff * 1.1 / 2
        camarilla_l4[i] = c - diff * 1.1 / 2
        camarilla_h3[i] = c + diff * 1.1 / 4
        camarilla_l3[i] = c - diff * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed bars only)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # === HTF: 1w data for weekly trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend filter
    ema_20w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Increased warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_20w_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # === Weekly Trend Filter ===
        weekly_uptrend = close[i] > ema_20w_aligned[i]
        weekly_downtrend = close[i] < ema_20w_aligned[i]
        
        # === Volume Confirmation ===
        volume_spike = vol_ratio[i] > 1.8
        
        # === Camarilla Level Conditions ===
        # Breakout conditions (continuation in trend direction)
        breakout_r4 = close[i] > h4_aligned[i]
        breakdown_s4 = close[i] < l4_aligned[i]
        
        # Fade conditions (mean reversion from extremes)
        fade_r3 = close[i] < h3_aligned[i]  # Price moved back below R3
        fade_s3 = close[i] > l3_aligned[i]  # Price moved back above S3
        
        # === Exit Logic (ATR-based stoploss) ===
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Exit conditions: opposite Camarilla level touch or middle line reversion
            if position_side > 0:  # Long exit
                if close[i] < l3_aligned[i] or close[i] < (h4_aligned[i] + l4_aligned[i]) / 2:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short exit
                if close[i] > h3_aligned[i] or close[i] > (h4_aligned[i] + l4_aligned[i]) / 2:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # === New Position Entry Logic (Only if Flat) ===
        # Long: 
        #   1) Breakout above R4 with volume + weekly uptrend (continuation)
        #   2) Fade below R3 with volume + weekly downtrend (mean reversion in downtrend)
        long_breakout = breakout_r4 and volume_spike and weekly_uptrend
        long_fade = fade_r3 and volume_spike and weekly_downtrend
        
        # Short:
        #   1) Breakdown below S4 with volume + weekly downtrend (continuation)
        #   2) Fade above S3 with volume + weekly uptrend (mean reversion in uptrend)
        short_breakout = breakdown_s4 and volume_spike and weekly_downtrend
        short_fade = fade_s3 and volume_spike and weekly_uptrend
        
        if long_breakout or long_fade:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout or short_fade:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals