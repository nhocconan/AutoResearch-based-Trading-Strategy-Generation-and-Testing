#!/usr/bin/env python3
"""
Experiment #3439: 6h Donchian Breakout + 12h Volume Spike + 1d Camarilla Pivot Fade
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts capture medium-term momentum. 
Fade entries at 1d Camarilla R3/S3 levels with volume confirmation (>2.0x 20-period average) 
provide high-probability mean reversion spots. Breakouts above R4/Below S4 with volume 
confirm continuation. 12h volume filter ensures institutional participation. 
Position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
Designed for both bull/bear markets: Camarilla levels adapt to volatility, volume confirms conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3439_6h_donchian20_12h_vol_1d_camarilla_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.ones(len(volume_12h))
    vol_ratio_12h[20:] = volume_12h[20:] / vol_ma_12h[20:]
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === 1d Indicators: Camarilla pivot levels (based on previous day) ===
    # Camarilla levels calculated from previous day's OHLC
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    # Calculate for each day, then align to 6h
    cam_r4_1d = np.full_like(close_1d, np.nan)
    cam_r3_1d = np.full_like(close_1d, np.nan)
    cam_s3_1d = np.full_like(close_1d, np.nan)
    cam_s4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):  # Start from 1 to use previous day
        pd_high = high_1d[i-1]
        pd_low = low_1d[i-1]
        pd_close = close_1d[i-1]
        cam_r4_1d[i] = pd_close + ((pd_high - pd_low) * 1.1 / 2)
        cam_r3_1d[i] = pd_close + ((pd_high - pd_low) * 1.1 / 4)
        cam_s3_1d[i] = pd_close - ((pd_high - pd_low) * 1.1 / 4)
        cam_s4_1d[i] = pd_close - ((pd_high - pd_low) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    cam_r4_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_r4_1d)
    cam_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_r3_1d)
    cam_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_s3_1d)
    cam_s4_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_s4_1d)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback, 20, 14, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(cam_r3_1d_aligned[i]) or np.isnan(cam_s3_1d_aligned[i]) or
            np.isnan(cam_r4_1d_aligned[i]) or np.isnan(cam_s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike on 12h (> 2.0x average) for confirmation
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        if volume_spike:
            # Camarilla fade logic: sell at R3, buy at S3
            # Camarilla breakout logic: buy above R4, sell below S4
            
            # Long entry conditions:
            # 1. Fade: price <= S3 and price >= S4 (between S3 and S4) 
            # 2. Breakout: price > R4
            if (price <= cam_s3_1d_aligned[i] and price >= cam_s4_1d_aligned[i]) or \
               (price > cam_r4_1d_aligned[i]):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry conditions:
            # 1. Fade: price >= R3 and price <= R4 (between R3 and R4)
            # 2. Breakout: price < S4
            elif (price >= cam_r3_1d_aligned[i] and price <= cam_r4_1d_aligned[i]) or \
                 (price < cam_s4_1d_aligned[i]):
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals