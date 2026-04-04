#!/usr/bin/env python3
"""
Experiment #4371: 6h Camarilla Pivot + 1d Volume Spike + ATR Regime Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) derived from 1d OHLC provide institutional support/resistance. Enter long at S3 with volume spike (>2x average) in ranging market (ATR ratio < 1.2), short at R3 with volume spike. Exit on R4/S4 breakout with volume. Uses discrete position sizing (0.25) to limit drawdown. Works in bull via R4 breakouts, in bear via S3 mean reversion and R4 breakdowns. Targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4371_6h_camarilla1d_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1d OHLC for Camarilla pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate Camarilla pivot levels from previous day
        # Based on yesterday's OHLC: H1 = C + 1.1*(H-L)/2, L1 = C - 1.1*(H-L)/2
        # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
        # R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
        prev_high = df_1d['high'].shift(1).values  # Previous day high
        prev_low = df_1d['low'].shift(1).values    # Previous day low
        prev_close = df_1d['close'].shift(1).values # Previous day close
        
        # Calculate pivot levels
        camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
        camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
        camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1
        camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1
        
        # Align to 6h timeframe (shifted by 1 day for completed bars only)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for regime filter ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # ATR ratio: current ATR / 50-period MA ATR (regime: <1.2 = low vol ranging, >1.2 = high vol trending)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.ones(n)
    atr_ratio[50:] = atr[50:] / atr_ma[50:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 50, 14)  # Vol MA, ATR MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long
                # Exit if price reaches R4 (breakout continuation) with volume
                if price >= camarilla_r4_aligned[i] and vol_ratio[i] > 1.5:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price falls below S3 (mean reversion failed)
                elif price < camarilla_s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if price reaches S4 (breakdown continuation) with volume
                if price <= camarilla_s4_aligned[i] and vol_ratio[i] > 1.5:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price rises above R3 (mean reversion failed)
                elif price > camarilla_r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: spike > 2x average
        volume_confirm = vol_ratio[i] > 2.0
        
        # Regime filter: only trade in low volatility ranging markets (ATR ratio < 1.2)
        ranging_market = atr_ratio[i] < 1.2
        
        # Long conditions: price at S3 support + volume spike + ranging market
        long_entry = (price <= camarilla_s3_aligned[i] * 1.001) and volume_confirm and ranging_market
        
        # Short conditions: price at R3 resistance + volume spike + ranging market
        short_entry = (price >= camarilla_r3_aligned[i] * 0.999) and volume_confirm and ranging_market
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals