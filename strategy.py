#!/usr/bin/env python3
"""
Experiment #275: 6h Elder Ray + 1d ADX Trend Filter + Volume Confirmation
HYPOTHESIS: Elder Ray (Bull Power/Bear Power) identifies institutional buying/selling pressure. 
Combined with 1d ADX (>25) to filter for trending markets and volume confirmation (>1.5x average) 
to ensure participation. Works in bull markets via strong Bull Power + ADX uptrend and in bear 
markets via strong Bear Power + ADX downtrend. Target: 75-150 total trades over 4 years (19-37/year).
Uses discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_275_6h_elder_ray_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX and EMA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d data
    # True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Movement
    up_move = df_1d['high'].diff().values
    down_move = -df_1d['low'].diff().values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # EMA(20) for trend direction
    ema_20 = pd.Series(df_1d['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # === 6h Indicators: Elder Ray Components ===
    # EMA(13) as reference
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema_13
    
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
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Enough for 20-period EMA and 14-period ATR/ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(ema_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- ADX Trend Filter: Require ADX > 25 for trending market ---
        strong_trend = adx_aligned[i] > 25
        
        # --- Elder Ray Logic ---
        # Bullish: Bull Power > 0 AND price > EMA(13) (confirmation)
        bullish = bull_power[i] > 0 and price > ema_13[i]
        # Bearish: Bear Power < 0 AND price < EMA(13) (confirmation)
        bearish = bear_power[i] < 0 and price < ema_13[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Bear Power turning negative with volume
                if bear_power[i] < 0 and volume_spike and not bullish:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Bull Power turning positive with volume
                if bull_power[i] > 0 and volume_spike and not bearish:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + strong trend + Elder Ray alignment
        if volume_spike and strong_trend:
            # Long: Bull Power positive AND EMA(20) rising (uptrend)
            if bullish and ema_20_aligned[i] > ema_20_aligned[max(0, i-1)]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Bear Power negative AND EMA(20) falling (downtrend)
            elif bearish and ema_20_aligned[i] < ema_20_aligned[max(0, i-1)]:
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