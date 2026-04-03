#!/usr/bin/env python3
"""
Experiment #187: 6h Camarilla Pivot + Volume Spike + ATR Regime Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from 1d timeframe act as institutional support/resistance. 
On 6h timeframe: 
- Long when price breaks above R4 with volume spike (>2.0x) AND ATR regime is expanding (ATR(6)/ATR(24) > 1.2) 
- Short when price breaks below S4 with volume spike AND ATR regime expanding 
- Exit when price retouches R3/S3 (mean reversion) or ATR contracts (<0.8) 
Volume confirms breakout strength, ATR regime filter avoids false breakouts in low-volatility environments. 
6h timeframe targets 12-37 trades/year (50-150 total over 4 years) with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_187_6h_camarilla_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivots from 1d OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    if len(df_1d) >= 1:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True range for 1d
        tr_1d = np.maximum(high_1d - low_1d, np.maximum(abs(high_1d - np.roll(close_1d, 1)), abs(low_1d - np.roll(close_1d, 1))))
        tr_1d[0] = high_1d[0] - low_1d[0]  # First bar
        
        # Camarilla levels: based on previous day's range
        for i in range(len(df_1d)):
            if i == 0:
                # Use first available data for warmup
                camarilla_r3[i] = camarilla_s3[i] = camarilla_r4[i] = camarilla_s4[i] = np.nan
                continue
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_ = prev_high - prev_low
            
            camarilla_r3[i] = prev_close + range_ * 1.1 / 4
            camarilla_s3[i] = prev_close - range_ * 1.1 / 4
            camarilla_r4[i] = prev_close + range_ * 1.1 / 2
            camarilla_s4[i] = prev_close - range_ * 1.1 / 2
        
        # Align to 6h timeframe (shifted by 1 for completed bars only)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        camarilla_r3_aligned = camarilla_s3_aligned = camarilla_r4_aligned = camarilla_s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(6) and ATR(24) for regime filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_6 = pd.Series(tr).ewm(span=6, min_periods=6, adjust=False).mean().values
    atr_24 = pd.Series(tr).ewm(span=24, min_periods=24, adjust=False).mean().values
    atr_ratio = np.zeros(n)
    atr_ratio[24:] = atr_6[24:] / atr_24[24:]
    atr_ratio[:24] = 1.0  # Neutral for warmup
    
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
    
    warmup = 50  # Ensure enough data for HTF pivots and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr_ratio[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- ATR Regime Filter: Only trade when volatility is expanding (>1.2) or contracting (<0.8) for exits ---
        vol_expanding = atr_ratio[i] > 1.2
        vol_contracting = atr_ratio[i] < 0.8
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Breakout Conditions ---
        breakout_r4 = close[i] > camarilla_r4_aligned[i]
        breakdown_s4 = close[i] < camarilla_s4_aligned[i]
        
        # --- Camarilla Mean Reversion Exits (touch R3/S3) ---
        retouch_r3 = abs(close[i] - camarilla_r3_aligned[i]) < (0.1 * camarilla_r4_aligned[i])  # Within 0.1% of R3
        retouch_s3 = abs(close[i] - camarilla_s3_aligned[i]) < (0.1 * camarilla_s4_aligned[i])  # Within 0.1% of S3
        
        # --- Exit Logic ---
        if in_position:
            # Exit on mean reversion to R3/S3 or volatility contraction
            if position_side > 0:  # Long position
                if retouch_r3 or vol_contracting:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if retouch_s3 or vol_contracting:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above R4 + volume spike + volatility expanding
        long_condition = breakout_r4 and volume_spike and vol_expanding
        
        # Short: Break below S4 + volume spike + volatility expanding
        short_condition = breakdown_s4 and volume_spike and vol_expanding
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals