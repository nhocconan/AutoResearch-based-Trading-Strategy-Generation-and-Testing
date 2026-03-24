#!/usr/bin/env python3
"""
Experiment #411: 6h Primary + 1w/1d HTF — Weekly Pivot + RSI Mean Reversion

Hypothesis: 6h timeframe (4 bars/day, 28 bars/week) uniquely aligns with weekly price cycles.
Weekly pivot levels (PP, R1, S1) act as natural S/R on 6h charts. Combined with RSI extremes
and 1d trend bias, this creates high-probability mean reversion setups.

Key insights from failed 6h experiments (#400, #403, #407, #410):
1. Complex regime detection (ADX+CHOP) caused 0 trades - SIMPLIFY
2. Too many confluence filters = no entries - use MAX 3 conditions
3. Weekly pivot levels underutilized on 6h - this is the EDGE

Entry Logic (SIMPLIFIED for trade generation):
- Long: Price near Weekly S1/S2 (within 1%) + RSI < 25 + 1d HMA bullish OR flat
- Short: Price near Weekly R1/R2 (within 1%) + RSI > 75 + 1d HMA bearish OR flat
- Breakout: Price breaks Weekly PP with volume + 1d HMA aligned

Why this should work on 6h:
- Weekly pivots are respected on 6h (institutional levels)
- RSI extremes on 6h = multi-day oversold/overbought (stronger signal)
- 1d HMA filters counter-trend trades in strong trends
- Fewer trades than 4h, more than 12h = optimal fee/trade balance

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_rsi_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_weekly_pivots(htf_high, htf_low, htf_close):
    """
    Calculate weekly pivot levels from HTF (1w) data.
    PP = (H + L + C) / 3
    R1 = 2*PP - L
    S1 = 2*PP - H
    R2 = PP + (H - L)
    S2 = PP - (H - L)
    """
    n = len(htf_close)
    pp = np.zeros(n)
    r1 = np.zeros(n)
    s1 = np.zeros(n)
    r2 = np.zeros(n)
    s2 = np.zeros(n)
    
    pp[:] = np.nan
    r1[:] = np.nan
    s1[:] = np.nan
    r2[:] = np.nan
    s2[:] = np.nan
    
    for i in range(1, n):
        h = htf_high[i-1]  # Previous week's high
        l = htf_low[i-1]   # Previous week's low
        c = htf_close[i-1] # Previous week's close
        
        pp[i] = (h + l + c) / 3.0
        r1[i] = 2.0 * pp[i] - l
        s1[i] = 2.0 * pp[i] - h
        r2[i] = pp[i] + (h - l)
        s2[i] = pp[i] - (h - l)
    
    return pp, r1, s1, r2, s2

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivots from 1w data
    htf_high_1w = df_1w['high'].values
    htf_low_1w = df_1w['low'].values
    htf_close_1w = df_1w['close'].values
    
    pp_1w_raw, r1_1w_raw, s1_1w_raw, r2_1w_raw, s2_1w_raw = calculate_weekly_pivots(
        htf_high_1w, htf_low_1w, htf_close_1w
    )
    
    # Align weekly pivots to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w_raw)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w_raw)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w_raw)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w_raw)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w_raw)
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(pp_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1D TREND BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 6H HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === WEEKLY PIVOT ZONES ===
        # Check if price is near support/resistance levels (within 1.5%)
        near_s1 = abs(close[i] - s1_aligned[i]) / close[i] < 0.015
        near_s2 = abs(close[i] - s2_aligned[i]) / close[i] < 0.015
        near_r1 = abs(close[i] - r1_aligned[i]) / close[i] < 0.015
        near_r2 = abs(close[i] - r2_aligned[i]) / close[i] < 0.015
        near_pp = abs(close[i] - pp_aligned[i]) / close[i] < 0.015
        
        # Price below pivot = bullish setup zone
        below_pp = close[i] < pp_aligned[i]
        above_pp = close[i] > pp_aligned[i]
        
        # === RSI EXTREMES (LOOSENED for more trades) ===
        rsi_oversold = rsi[i] < 30.0
        rsi_overbought = rsi[i] > 70.0
        rsi_extreme_oversold = rsi[i] < 20.0
        rsi_extreme_overbought = rsi[i] > 80.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = False
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 1.1 * vol_sma[i]
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50[i] if not np.isnan(sma_50[i]) else False
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === ENTRY LOGIC (SIMPLIFIED - ensure trades) ===
        desired_signal = 0.0
        
        # LONG SETUP: Near weekly support + RSI oversold + trend filter
        if rsi_oversold and (near_s1 or near_s2 or below_pp):
            # Require at least ONE of: 1d bull, 6h bull, or above SMA200
            trend_ok = htf_1d_bull or hma_bull or above_sma200
            if trend_ok:
                if rsi_extreme_oversold or vol_confirm:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT SETUP: Near weekly resistance + RSI overbought + trend filter
        elif rsi_overbought and (near_r1 or near_r2 or above_pp):
            # Require at least ONE of: 1d bear, 6h bear, or below SMA200
            trend_ok = htf_1d_bear or hma_bear or (not above_sma200)
            if trend_ok:
                if rsi_extreme_overbought or vol_confirm:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # BREAKOUT LONG: Price breaks above PP with volume + 1d bull
        elif near_pp and above_pp and htf_1d_bull and vol_confirm and rsi[i] > 45:
            desired_signal = SIZE_BASE
        
        # BREAKOUT SHORT: Price breaks below PP with volume + 1d bear
        elif near_pp and below_pp and htf_1d_bear and vol_confirm and rsi[i] < 55:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals