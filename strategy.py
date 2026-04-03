#!/usr/bin/env python3
"""
Experiment #699: 6h Camarilla Pivot Breakout + 12h Volume Spike + 1d ADX Trend Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from daily timeframe act as institutional support/resistance. 
Breakout confirmed by 12h volume spike (>2.0x) and filtered by 1d ADX > 25 ensures we only trade in trending markets. 
This avoids false breakouts in ranging markets. Discrete position sizing (0.25) minimizes fee churn. 
Works in bull/bear markets via ADX trend filter: long on upside breakouts, short on downside breakouts only when trending.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_699_6h_camarilla_pivot_12h_vol_1d_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and ADX (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Camarilla: based on previous day's range
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_s4 = np.zeros(len(close_1d))
    camarilla_r3 = np.zeros(len(close_1d))
    camarilla_r4 = np.zeros(len(close_1d))
    camarilla_pivot = np.zeros(len(close_1d))
    
    for i in range(1, len(high_1d)):
        # Previous day's OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        # Pivot point (standard)
        camarilla_pivot[i] = (phigh + plow + pclose) / 3
        
        # Range
        rang = phigh - plow
        
        # Camarilla levels
        camarilla_s3[i] = pclose - rang * 1.1 / 4
        camarilla_s4[i] = pclose - rang * 1.1 / 2
        camarilla_r3[i] = pclose + rang * 1.1 / 4
        camarilla_r4[i] = pclose + rang * 1.1 / 2
    
    # For first bar, use same day's data (will be overridden by alignment shift)
    camarilla_s3[0] = camarilla_s3[1] if len(camarilla_s3) > 1 else close_1d[0]
    camarilla_s4[0] = camarilla_s4[1] if len(camarilla_s4) > 1 else close_1d[0]
    camarilla_r3[0] = camarilla_r3[1] if len(camarilla_r3) > 1 else close_1d[0]
    camarilla_r4[0] = camarilla_r4[1] if len(camarilla_r4) > 1 else close_1d[0]
    camarilla_pivot[0] = camarilla_pivot[1] if len(camarilla_pivot) > 1 else close_1d[0]
    
    # Calculate ADX for 1d timeframe
    adx_period = 14
    
    # True Range
    tr_1d = np.zeros(len(high_1d))
    for i in range(1, len(high_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    dm_plus = np.zeros(len(high_1d))
    dm_minus = np.zeros(len(high_1d))
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
        dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed TR, DM+
    tr_smooth = pd.Series(tr_1d).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = np.zeros(len(di_plus))
    for i in range(len(di_plus)):
        if di_plus[i] + di_minus[i] != 0:
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        else:
            dx[i] = 0
    
    adx_1d = pd.Series(dx).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
    
    # Align HTF indicators to 6h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 12h data for volume confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Volume MA(20) for 12h timeframe
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.ones(len(volume_12h))
    vol_ratio_12h[20:] = volume_12h[20:] / vol_ma_12h[20:]
    
    # Align volume ratio to 6h timeframe
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average on 12h) ---
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        # --- Trend Filter: Require ADX > 25 (trending market) ---
        trending = adx_1d_aligned[i] > 25
        
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
            
            # Optional: time-based exit after 6 bars (~36h on 6h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike and trending:
            # Long: Price breaks above R4 with volume
            if price > camarilla_r4_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Price breaks below S4 with volume
            elif price < camarilla_s4_aligned[i]:
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