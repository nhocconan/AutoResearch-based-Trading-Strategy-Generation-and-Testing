#!/usr/bin/env python3
"""
Experiment #003: 4h Camarilla + Volume + Choppiness (Simplified)

HYPOTHESIS: Camarilla pivot zones from 1d provide institutional support/resistance.
Volume spike confirms the level is significant. Choppiness filters choppy periods.
Simple = fewer trades = less fee drag = better Sharpe.

KEY INSIGHT: The DB reference (Sharpe=1.471) succeeded with SIMPLE conditions.
Too many filters = too few trades = negative Sharpe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_simple_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[tr[0]], tr])
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - trending < 53, ranging > 53"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    chop = np.full(n, np.nan)
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        prange = hh - ll
        if prange > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / prange) / np.log10(period)
    return chop

def calculate_ema(close, span):
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # 12h EMA for trend direction
    df_12h = get_htf_data(prices, '12h')
    ema_12h_raw = calculate_ema(df_12h['close'].values, 21)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_raw)
    
    # Camarilla pivots from 1d
    pivots_h = df_1d['high'].values
    pivots_l = df_1d['low'].values
    pivots_c = df_1d['close'].values
    
    # Calculate S3, R3 levels (the most effective Camarilla levels)
    rng = pivots_h - pivots_l
    s3_raw = pivots_c - rng * 1.1 / 4
    r3_raw = pivots_c + rng * 1.1 / 4
    s4_raw = pivots_c - rng * 1.1 / 2
    r4_raw = pivots_c + rng * 1.1 / 2
    
    # Align to 4h
    s3 = align_htf_to_ltf(prices, df_1d, s3_raw)
    r3 = align_htf_to_ltf(prices, df_1d, r3_raw)
    s4 = align_htf_to_ltf(prices, df_1d, s4_raw)
    r4 = align_htf_to_ltf(prices, df_1d, r4_raw)
    
    # 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_pnl = 0.0
    lowest_pnl = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0 or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Skip if pivot levels not ready
        if np.isnan(s3[i]) or np.isnan(r3[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        is_trending = chop_14[i] < 53.0  # Only trade in trending regime
        
        # === TREND DIRECTION ===
        bull_trend = close[i] > ema_12h_aligned[i] if not np.isnan(ema_12h_aligned[i]) else True
        bear_trend = close[i] < ema_12h_aligned[i] if not np.isnan(ema_12h_aligned[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # Distance to pivot levels in ATR units
        atr = atr_14[i]
        dist_s3 = (close[i] - s3[i]) / atr
        dist_r3 = (r3[i] - close[i]) / atr
        dist_s4 = (close[i] - s4[i]) / atr if not np.isnan(s4[i]) else 999
        dist_r4 = (r4[i] - close[i]) / atr if not np.isnan(r4[i]) else 999
        
        # === SIMPLE ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trending:
            # LONG: Price bouncing from S3 or S4 support with volume
            if dist_s3 > -0.3 and dist_s3 < 1.5 and bull_trend and vol_spike:
                desired_signal = SIZE
            elif dist_s4 > -0.3 and dist_s4 < 1.5 and bull_trend and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Price bouncing from R3 or R4 resistance with volume
            if dist_r3 > -0.3 and dist_r3 < 1.5 and bear_trend and vol_spike:
                desired_signal = -SIZE
            elif dist_r4 > -0.3 and dist_r4 < 1.5 and bear_trend and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS ===
        if in_position and position_side > 0:
            if low[i] < stop_price:
                desired_signal = 0.0
            else:
                # Trailing stop
                trailing = high[i] - 2.0 * atr
                stop_price = max(stop_price, trailing)
                if low[i] < stop_price:
                    desired_signal = 0.0
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                desired_signal = 0.0
            else:
                # Trailing stop
                trailing = low[i] + 2.0 * atr
                stop_price = min(stop_price, trailing)
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New entry or reversal
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr
                if position_side > 0:
                    stop_price = entry_price - 2.0 * atr
                else:
                    stop_price = entry_price + 2.0 * atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals