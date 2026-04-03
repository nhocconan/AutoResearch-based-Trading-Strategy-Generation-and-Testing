#!/usr/bin/env python3
"""
Experiment #819: 6h Camarilla Pivot + 12h Trend Filter + Volume Spike
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
provide high-probability entry zones. In ranging markets (ADX<25), fade extremes 
at R3/S3. In trending markets (ADX>25), breakout continuation at R4/S4. 
12h EMA(21) trend filter ensures alignment with higher timeframe momentum. 
Volume spike (>2.0x average) confirms participation. Discrete sizing 0.25. 
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_819_6h_camarilla12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(21) on 12h
    ema_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    # Trend: 1 = rising (price > EMA), -1 = falling (price < EMA), 0 = neutral
    ema_trend_12h = np.zeros_like(ema_12h)
    ema_trend_12h[21:] = np.where(close_12h[21:] > ema_12h[21:], 1, 
                                  np.where(close_12h[21:] < ema_12h[21:], -1, 0))
    # Align trend to 6h timeframe
    ema_trend_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_trend_12h)
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    # Camarilla levels calculated from previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 4.0)
    s3 = pivot - (range_val * 1.1 / 4.0)
    r4 = pivot + (range_val * 1.1 / 2.0)
    s4 = pivot - (range_val * 1.1 / 2.0)
    
    # === 6h Indicators: ADX(14) for regime detection ===
    def calculate_atr(high, low, close, period):
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    tr_smooth = pd.Series(atr).ewm(span=14, min_periods=14, adjust=False).mean().values  # ATR is already smoothed
    
    # DI and DX
    plus_di = np.where(tr_smooth > 0, 100 * plus_dm_smooth / tr_smooth, 0)
    minus_di = np.where(tr_smooth > 0, 100 * minus_dm_smooth / tr_smooth, 0)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Regime: 1 = trending (ADX>25), 0 = ranging (ADX<=25)
    regime = np.where(adx > 25, 1, 0)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 21)  # sufficient for volume MA, EMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or
            np.isnan(ema_trend_12h_aligned[i]) or np.isnan(regime[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~32h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            if regime[i] == 0:  # Ranging market (ADX <= 25): mean reversion at R3/S3
                # Long: price < S3 and 12h EMA trending up (bullish bias)
                if price < s3[i] and ema_trend_12h_aligned[i] > 0:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: price > R3 and 12h EMA trending down (bearish bias)
                elif price > r3[i] and ema_trend_12h_aligned[i] < 0:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:  # Trending market (ADX > 25): breakout continuation at R4/S4
                # Long: price > R4 and 12h EMA trending up
                if price > r4[i] and ema_trend_12h_aligned[i] > 0:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: price < S4 and 12h EMA trending down
                elif price < s4[i] and ema_trend_12h_aligned[i] < 0:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals