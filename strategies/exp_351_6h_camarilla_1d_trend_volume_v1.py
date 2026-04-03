#!/usr/bin/env python3
"""
Experiment #351: 6h Camarilla Pivot + 1d Trend + Volume Spike (Revised)

HYPOTHESIS: Camarilla pivot levels from 1d provide intraday support/resistance zones. 
Breakouts above R4 or below S4 with volume confirmation (>1.5x average) and aligned 
1d trend (close > 1d EMA50) capture strong momentum moves. Fade at R3/S3 in ranging 
markets (1d ADX < 25). 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) 
to minimize fee drag. Works in bull (breakouts with volume) and bear (failed reversals 
at R3/S3) markets. This version fixes entry logic and adds minimum holding period to 
increase trade frequency from previous near-zero attempts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_351_6h_camarilla_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    def calculate_camarilla(h, l, c):
        """Calculate Camarilla pivot levels: R4, R3, R2, R1, PP, S1, S2, S3, S4"""
        range_ = h - l
        pp = (h + l + c) / 3.0
        r4 = c + range_ * 1.1 / 2.0
        r3 = c + range_ * 1.1 / 4.0
        r2 = c + range_ * 1.1 / 6.0
        r1 = c + range_ * 1.1 / 12.0
        s1 = c - range_ * 1.1 / 12.0
        s2 = c - range_ * 1.1 / 6.0
        s3 = c - range_ * 1.1 / 4.0
        s4 = c - range_ * 1.1 / 2.0
        return r4, r3, r2, r1, pp, s1, s2, s3, s4
    
    # Calculate for each 1d bar
    r4_1d = np.full(len(df_1d), np.nan)
    r3_1d = np.full(len(df_1d), np.nan)
    s3_1d = np.full(len(df_1d), np.nan)
    s4_1d = np.full(len(df_1d), np.nan)
    pp_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        r4, r3, r2, r1, pp, s1, s2, s3, s4 = calculate_camarilla(
            df_1d['high'].iloc[i], 
            df_1d['low'].iloc[i], 
            df_1d['close'].iloc[i]
        )
        r4_1d[i] = r4
        r3_1d[i] = r3
        s3_1d[i] = s3
        s4_1d[i] = s4
        pp_1d[i] = pp
    
    # Align Camarilla levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d ADX for regime detection (trending vs ranging)
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
    
    warmup = 100  # Warmup for 1d indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d Regime Filter: ADX > 25 = trending, ADX < 25 = ranging ---
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 25
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price Levels ---
        price = close[i]
        r4 = r4_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        ema50 = ema50_1d_aligned[i]
        pp = pp_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
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
                # Exit on mean reversion to pivot point in ranging markets
                if is_ranging and abs(price - pp) < 0.5 * atr_14[i]:
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
                # Exit on mean reversion to pivot point in ranging markets
                if is_ranging and abs(price - pp) < 0.5 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: Price > R4 + volume spike + 1d trend up (price > EMA50)
        long_breakout = (price > r4) and volume_spike and (price > ema50)
        
        # Short breakout: Price < S4 + volume spike + 1d trend down (price < EMA50)
        short_breakout = (price < s4) and volume_spike and (price < ema50)
        
        # Long fade: Price < R3 + volume spike + ranging market (fade from resistance)
        long_fade = (price < r3) and volume_spike and is_ranging and (price > pp)
        
        # Short fade: Price > S3 + volume spike + ranging market (fade from support)
        short_fade = (price > s3) and volume_spike and is_ranging and (price < pp)
        
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