#!/usr/bin/env python3
"""
Experiment #111: 6h Camarilla Pivot + Volume Spike + Daily Trend Filter

HYPOTHESIS: Camarilla pivot levels (calculated from 1d OHLC) act as intraday support/resistance.
At 6h timeframe: fade at R3/S3 levels (mean reversion in range), breakout continuation at R4/S4 levels (trend).
Uses 1d EMA50 as trend filter to avoid fading in strong trends. Volume spike (2x 20-period MA) confirms participation.
Designed for 50-150 total trades over 4 years (12-37/year) to avoid overtrading. Works in bull/bear by switching
between mean reversion (fade R3/S3) and trend following (break R4/S4) based on 1d EMA50 slope.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_pivot_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot calculation and trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate Camarilla levels from previous 1d bar (shifted to avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    r3 = pivot + (range_ * 1.1 / 2)
    s3 = pivot - (range_ * 1.1 / 2)
    r4 = pivot + (range_ * 1.1)
    s4 = pivot - (range_ * 1.1)
    
    # Align HTF arrays to LTF (6h) with shift(1) for completed bars only
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    # Trend: slope of EMA50 over 3 periods (approx 3d)
    ema_50_slope = np.zeros_like(ema_50_aligned)
    ema_50_slope[3:] = (ema_50_aligned[3:] - ema_50_aligned[:-3]) / 3
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_slope)
    
    # === 6h Indicators ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Price relative to Camarilla levels ---
        price = close[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        
        # --- Trend filter: EMA50 slope ---
        trend_up = ema_50_slope_aligned[i] > 0
        trend_down = ema_50_slope_aligned[i] < 0
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit conditions:
                # 1. Mean reversion: price reaches opposite S3 (if fading R3)
                # 2. Trend exhaustion: price crosses below EMA50 in strong uptrend
                # 3. Stoploss: 2.5 * ATR(14) (simplified as 2.5% of price for now)
                if price <= s3 or (trend_up and price < ema_50_aligned[i]):
                    exit_signal = True
            else:  # Short position
                # Exit conditions:
                # 1. Mean reversion: price reaches opposite R3 (if fading S3)
                # 2. Trend exhaustion: price crosses above EMA50 in strong downtrend
                if price >= r3 or (trend_down and price > ema_50_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Fade at R3/S3 in ranging market (low EMA50 slope magnitude)
        # Breakout at R4/S4 in trending market (high EMA50 slope magnitude)
        slope_mag = abs(ema_50_slope_aligned[i])
        ranging = slope_mag < 0.0001  # low threshold for 6h EMA50 slope
        trending = slope_mag >= 0.0001
        
        if ranging and vol_ok:
            # Fade at R3/S3: sell at R3, buy at S3
            if price >= r3:
                # Sell at R3 (expect mean reversion down)
                in_position = True
                position_side = -1
                entry_bar = i
                signals[i] = -SIZE
            elif price <= s3:
                # Buy at S3 (expect mean reversion up)
                in_position = True
                position_side = 1
                entry_bar = i
                signals[i] = SIZE
        elif trending and vol_ok:
            # Breakout at R4/S4: buy break above R4, sell break below S4
            if price > r4 and trend_up:
                # Buy breakout above R4 in uptrend
                in_position = True
                position_side = 1
                entry_bar = i
                signals[i] = SIZE
            elif price < s4 and trend_down:
                # Sell breakdown below S4 in downtrend
                in_position = True
                position_side = -1
                entry_bar = i
                signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals