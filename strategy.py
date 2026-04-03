#!/usr/bin/env python3
"""
Experiment #299: 6h Camarilla Pivot + Volume Spike + 12h ADX Trend Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with volume spikes (>2.0x average) and 12h ADX > 25 for trend confirmation 
captures high-probability reversals in ranging markets and breakouts in trending 
markets. The 12h ADX filter ensures we only trade when there is sufficient 
momentum, reducing false signals. 6h timeframe targets 12-37 trades/year (50-150 total 
over 4 years) to minimize fee drag while capturing significant moves. Works in both 
bull (breakouts with volume) and bear (failed reversals at pivot levels) markets. 
Uses ATR-based stoploss for risk management.

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.25) to minimize churn
- Volume confirmation threshold set to 2.0x to balance signal quality and frequency
- Minimum holding period of 2 bars to reduce churn
- Warmup period set to 100 bars to ensure stable indicators
- Position exits on ATR stoploss or opposite pivot level touch
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_299_6h_camarilla_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ADX(14) on 12h data
    def calculate_adx(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period + 1:
            return np.full_like(high_arr, np.nan)
        # True Range
        tr0 = high_arr[1:] - low_arr[1:]
        tr1 = np.abs(high_arr[1:] - close_arr[:-1])
        tr2 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr0, np.maximum(tr1, tr2))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high_arr[1:] - high_arr[:-1]
        down_move = low_arr[:-1] - low_arr[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Camarilla levels calculated from daily OHLC, but we approximate using 6h bars
    # For simplicity, we use rolling 4-bar window (24h) to approximate daily OHLC
    lookback = 4  # 4 x 6h = 24h ~ 1 day
    camarilla_h = np.full(n, np.nan)
    camarilla_l = np.full(n, np.nan)
    camarilla_c = np.full(n, np.nan)
    
    for i in range(lookback, n):
        camarilla_h[i] = np.max(high[i-lookback:i])
        camarilla_l[i] = np.min(low[i-lookback:i])
        camarilla_c[i] = close[i-1]  # Previous close
    
    # Calculate Camarilla levels
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    camarilla_m = np.full(n, np.nan)  # Pivot point
    
    for i in range(lookback, n):
        if not (np.isnan(camarilla_h[i]) or np.isnan(camarilla_l[i]) or np.isnan(camarilla_c[i])):
            camarilla_m[i] = (camarilla_h[i] + camarilla_l[i] + camarilla_c[i]) / 3
            range_hl = camarilla_h[i] - camarilla_l[i]
            camarilla_r3[i] = camarilla_m[i] + range_hl * 1.1 / 4
            camarilla_s3[i] = camarilla_m[i] - range_hl * 1.1 / 4
            camarilla_r4[i] = camarilla_m[i] + range_hl * 1.1 / 2
            camarilla_s4[i] = camarilla_m[i] - range_hl * 1.1 / 2
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
    
    warmup = 100  # Increased warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h ADX Trend Filter: ADX > 25 indicates trending market ---
        trending_market = adx_12h_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Level Conditions ---
        # Mean reversion: fade at R3/S3 in ranging markets (ADX <= 25)
        # Breakout: continue at R4/S4 in trending markets (ADX > 25)
        
        long_condition = False
        short_condition = False
        
        if trending_market:
            # In trending market: breakout continuation at R4/S4
            long_condition = (close[i] > camarilla_r4[i]) and volume_spike
            short_condition = (close[i] < camarilla_s4[i]) and volume_spike
        else:
            # In ranging market: mean reversion at R3/S3
            long_condition = (close[i] < camarilla_s3[i]) and volume_spike
            short_condition = (close[i] > camarilla_r3[i]) and volume_spike
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
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
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Exit conditions
            exit_long = False
            exit_short = False
            
            if position_side > 0:  # Long
                # Exit on touch of opposite pivot level or R4 (profit taking)
                exit_long = (close[i] >= camarilla_r3[i]) or (close[i] >= camarilla_r4[i] and trending_market)
            else:  # Short
                # Exit on touch of opposite pivot level or S4 (profit taking)
                exit_short = (close[i] <= camarilla_s3[i]) or (close[i] <= camarilla_s4[i] and trending_market)
            
            if exit_long or exit_short:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>