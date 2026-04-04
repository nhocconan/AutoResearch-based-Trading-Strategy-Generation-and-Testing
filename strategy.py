#!/usr/bin/env python3
"""
Experiment #3451: 6h Camarilla Pivot + 1d Trend + Volume Spike
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with 1d EMA trend filter and volume confirmation provides high-probability 
entries in both bull and bear markets. The 6h timeframe reduces trade frequency to 
avoid fee drag while Camarilla levels provide mathematical support/resistance that 
works across regimes. Volume spike confirms institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3451_6h_camarilla_pivot_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend and Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We calculate these from the COMPLETED 1d bar (hence shift(1) in align)
    cam_r4 = np.full(n, np.nan)
    cam_r3 = np.full(n, np.nan)
    cam_s3 = np.full(n, np.nan)
    cam_s4 = np.full(n, np.nan)
    
    # Only calculate when we have complete 1d data
    if len(close_1d) >= 2:
        # Use previous day's OHLC to calculate today's Camarilla levels
        prev_close = close_1d[:-1]  # yesterdays close
        prev_high = high_1d[:-1]    # yesterdays high
        prev_low = low_1d[:-1]      # yesterdays low
        
        # Camarilla calculations
        rang = prev_high - prev_low
        cam_r4_y = prev_close + (rang * 1.1 / 2)
        cam_r3_y = prev_close + (rang * 1.1 / 4)
        cam_s3_y = prev_close - (rang * 1.1 / 4)
        cam_s4_y = prev_close - (rang * 1.1 / 2)
        
        # Align to 6h timeframe (these levels are valid for the entire 6h period)
        cam_r4[:-1] = align_htf_to_ltf(prices, df_1d[:-1], cam_r4_y)  # exclude last incomplete bar
        cam_r3[:-1] = align_htf_to_ltf(prices, df_1d[:-1], cam_r3_y)
        cam_s3[:-1] = align_htf_to_ltf(prices, df_1d[:-1], cam_s3_y)
        cam_s4[:-1] = align_htf_to_ltf(prices, df_1d[:-1], cam_s4_y)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_loss = 0.0
    
    warmup = max(50, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(cam_r3[i]) or np.isnan(cam_s3[i]) or
            np.isnan(cam_r4[i]) or np.isnan(cam_s4[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Stoploss exit
            if position_side > 0:  # Long
                if price <= stop_loss:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price >= stop_loss:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # 1d EMA trend filter
            price_vs_ema = price - ema_1d_aligned[i]
            
            # Mean reversion entries at R3/S3 (fade extreme levels)
            # Long when price reaches S3 support in uptrend
            if price <= cam_s3[i] and price_vs_ema > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_loss = cam_s4[i]  # Stop below S4
                signals[i] = SIZE
            # Short when price reaches R3 resistance in downtrend
            elif price >= cam_r3[i] and price_vs_ema < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_loss = cam_r4[i]  # Stop above R4
                signals[i] = -SIZE
            # Breakout entries at R4/S4 (continuation)
            # Long breakout above R4 in uptrend
            elif price > cam_r4[i] and price_vs_ema > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_loss = cam_r3[i]  # Stop below R3
                signals[i] = SIZE
            # Short breakdown below S4 in downtrend
            elif price < cam_s4[i] and price_vs_ema < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_loss = cam_s3[i]  # Stop above S3
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals