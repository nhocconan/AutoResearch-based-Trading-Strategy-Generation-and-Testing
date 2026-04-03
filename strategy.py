#!/usr/bin/env python3
"""
Experiment #012: 12h Donchian(20) Breakout + 1d Trend + Volume Spike + Choppiness Filter

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, when aligned with 1d trend (price above/below 1d EMA50) and confirmed by volume spikes (>1.8x 20-bar MA) and low choppiness (CHOP(14) < 38.2 = trending), capture strong trending moves in both bull and bear markets. Uses ATR-based trailing stop (2.5x ATR). Designed for low trade frequency (~12-37/year) to minimize fee drag by requiring confluence of breakout, 1d trend, volume confirmation, and trending regime.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_trend_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_chop(high, low, close, period=14):
    """Choppiness Index: values > 61.8 = ranging, < 38.2 = trending."""
    n = len(close)
    if n < period * 2:
        return np.full(n, 50.0)
    
    atr_sum = np.zeros(n)
    for i in range(period, n):
        tr = 0
        for j in range(i - period + 1, i + 1):
            tr += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        atr_sum[i] = tr
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl < 1e-10, 1e-10, range_hl)
    
    log_sum = np.log10(atr_sum / range_hl) / np.log10(period)
    chop = 100 * log_sum
    
    # For warmup period, set to neutral
    chop[:period] = 50.0
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === HTF: 1w for additional regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # === 12h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_chop(high, low, close, period=14)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 200  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(chop_14[i]) or np.isnan(dc_upper_20[i]) or 
            np.isnan(dc_lower_20[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Levels ---
        ema_50_1d = ema_50_1d_aligned[i]
        ema_200_1w = ema_200_1w_aligned[i]
        
        # --- 1d Trend Filter ---
        trend_bullish = close[i] > ema_50_1d
        trend_bearish = close[i] < ema_50_1d
        
        # --- 1w Regime Filter (Bull/Bear Market) ---
        # In bull market (price above 200w EMA), favor longs
        # In bear market (price below 200w EMA), favor shorts
        bull_market = close[i] > ema_200_1w
        bear_market = close[i] < ema_200_1w
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.8 if vol_ma_20[i] > 1e-10 else False  # 1.8x volume spike
        
        # --- Choppiness Regime Filter ---
        # CHOP < 38.2 = trending (favor breakouts)
        # CHOP > 61.8 = ranging (avoid breakouts)
        trending_regime = chop_14[i] < 38.2
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or choppy regime
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~1d)
            if min_hold:
                if position_side > 0:
                    # Exit long: trend turns bearish OR market turns choppy
                    if trend_bearish or chop_14[i] > 50.0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: trend turns bullish OR market turns choppy
                    if trend_bullish or chop_14[i] > 50.0:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: Breakout above DC upper with bullish 1d trend AND volume confirmation AND trending regime
        # In bull market, longs are favored; in bear market, require stronger confirmation
        if bullish_breakout and trend_bullish and vol_ok and trending_regime:
            # In bear market, require additional confirmation (strong volume)
            if bear_market and volume[i] <= vol_ma_20[i] * 2.5:
                signals[i] = 0.0
                continue
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions: Breakout below DC lower with bearish 1d trend AND volume confirmation AND trending regime
        # In bear market, shorts are favored; in bull market, require stronger confirmation
        elif bearish_breakout and trend_bearish and vol_ok and trending_regime:
            # In bull market, require additional confirmation (strong volume)
            if bull_market and volume[i] <= vol_ma_20[i] * 2.5:
                signals[i] = 0.0
                continue
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals