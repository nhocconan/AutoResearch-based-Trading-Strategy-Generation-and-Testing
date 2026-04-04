#!/usr/bin/env python3
"""
Experiment #2491: 6h Camarilla Pivot Reversal + Volume Spike
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from 1d timeframe
provide institutional support/resistance. At 6h, we fade extreme touches of R3/S3 with volume confirmation
in ranging markets, and breakout continuation at R4/S4 with volume in trending markets. Uses discrete
sizing (0.25) to limit fee drift. Works in bull/bear via regime filter (ADX < 25 = range, > 25 = trend).
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2491_6h_camarilla_pivot_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla levels and ADX regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d (based on previous day)
    # Camarilla: H/L/C from previous period
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # True range approximation for volatility
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - prev_close),
                               np.abs(low_1d - prev_close)))
    atr_1d = pd.Series(tr).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    # Camarilla levels: H/L/C +- (H-L) * multipliers
    range_1d = prev_high - prev_low
    r3 = prev_close + range_1d * 1.1 / 4
    s3 = prev_close - range_1d * 1.1 / 4
    r4 = prev_close + range_1d * 1.1 / 2
    s4 = prev_close - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # ADX for regime detection (1d)
    # +DI, -DI, DX
    up_move = np.diff(high_1d, prepend=np.nan)
    down_move = -np.diff(low_1d, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / tr_14
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_6h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(adx_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss at 2*ATR(6h) equivalent ---
        # Use 6h Donchian width as ATR proxy
        if i >= 20:
            highest_20 = np.max(high[i-19:i+1])
            lowest_20 = np.min(low[i-19:i+1])
            atr_estimate = (highest_20 - lowest_20) * 0.15
        else:
            atr_estimate = price * 0.02  # fallback 2%
        
        if in_position:
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- Regime Filter: ADX < 25 = range (mean revert), > 25 = trend (breakout) ---
        is_ranging = adx_6h[i] < 25
        is_trending = adx_6h[i] >= 25
        
        # Volume confirmation: require spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if not volume_spike:
            signals[i] = 0.0
            continue
        
        # --- Entry Logic ---
        if is_ranging:
            # Mean reversion at R3/S3
            # Long: price touches/slightly breaks S3 then reverses up
            if price <= s3_6h[i] * 1.002 and price > s3_6h[i] * 0.998:  # near S3
                # Confirm reversal: close > open (bullish candle)
                if i > 0 and close[i] > prices["open"].iloc[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    signals[i] = SIZE
            # Short: price touches/slightly breaks R3 then reverses down
            elif price >= r3_6h[i] * 0.998 and price <= r3_6h[i] * 1.002:  # near R3
                # Confirm reversal: close < open (bearish candle)
                if i > 0 and close[i] < prices["open"].iloc[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    signals[i] = -SIZE
        else:  # is_trending
            # Breakout continuation at R4/S4
            # Long: price breaks above R4 with volume
            if price > r4_6h[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short: price breaks below S4 with volume
            elif price < s4_6h[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
    
    return signals