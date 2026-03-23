#!/usr/bin/env python3
"""
Experiment #061: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + Fisher Transform + Vol Expansion

Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts better to crypto regime changes than HMA/EMA.
Combined with Ehlers Fisher Transform for reversal timing and ATR vol expansion filter, this should
generate 25-50 trades/year with Sharpe > 0.486 on 4h timeframe.

Why this should work (learning from 60 failed experiments):
1) KAMA adapts ER (Efficiency Ratio) - smooth in trends, responsive in ranges (better than fixed HMA)
2) Fisher Transform normalizes price to Gaussian - cleaner reversal signals than RSI
3) ATR vol expansion filter - only trade when volatility is expanding (avoids chop)
4) 1d HTF for macro bias - prevents counter-trend entries in strong bear/bull markets
5) Simpler entry conditions - avoid over-filtering that caused 0 trades in exp #052, #055, #058, #060

Key differences from failed experiments:
- NO Choppiness Index (caused whipsaws in exp #049, #054, #056)
- NO CRSI (failed in exp #050, #054, #057, #060)
- NO volume filters (disaster in exp #059 Sharpe=-4.465)
- Fewer confluence requirements (ensure trades happen on all symbols)

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 25-50 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_vol_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - smooth in noise, fast in trends.
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[0] = 0
    volatility = np.zeros(len(close))
    for i in range(period, len(close)):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = np.zeros(len(close))
    er[period:] = change[period:] / (volatility[period:] + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for cleaner reversal signals.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - LL) / (HH - LL) - 0.67
    """
    close_s = pd.Series(close)
    
    # Normalize price to 0-1 range
    hh = close_s.rolling(window=period, min_periods=period).max()
    ll = close_s.rolling(window=period, min_periods=period).min()
    
    # Calculate X (normalized price shifted to -1 to +1)
    x = 0.67 * (close - ll.values) / (hh.values - ll.values + 1e-10) - 0.67
    x = np.clip(x, -0.99, 0.99)  # Prevent division by zero in log
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_atr_ratio(atr, period_short=7, period_long=30):
    """Calculate ATR ratio for volatility expansion detection."""
    atr_s = pd.Series(atr)
    atr_short = atr_s.ewm(span=period_short, min_periods=period_short, adjust=False).mean().values
    atr_long = atr_s.ewm(span=period_long, min_periods=period_long, adjust=False).mean().values
    
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d KAMA for intermediate trend
    kama_1d = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1w KAMA for macro trend
    kama_1w = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_4h = calculate_kama(close, period=10)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    atr_ratio = calculate_atr_ratio(atr_14, period_short=7, period_long=30)
    sma_50 = calculate_sma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(kama_4h[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(atr_ratio[i]) or np.isnan(sma_50[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_kama_1w = close[i] > kama_1w_aligned[i]
        price_below_kama_1w = close[i] < kama_1w_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === 4H KAMA SLOPE ===
        kama_slope_up = kama_4h[i] > kama_4h[i-3] if i > 3 else False
        kama_slope_down = kama_4h[i] < kama_4h[i-3] if i > 3 else False
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # Alternative: Fisher crossover with signal line
        fisher_cross_up = (fisher[i] > fisher_signal[i]) and (fisher[i-1] <= fisher_signal[i-1]) if i > 0 else False
        fisher_cross_down = (fisher[i] < fisher_signal[i]) and (fisher[i-1] >= fisher_signal[i-1]) if i > 0 else False
        
        # === VOLATILITY EXPANSION FILTER ===
        vol_expanding = atr_ratio[i] > 1.15  # Short-term ATR > 15% above long-term
        
        # === KAMA POSITION ===
        price_above_kama_4h = close[i] > kama_4h[i]
        price_below_kama_4h = close[i] < kama_4h[i]
        
        # === SMA50 FILTER (avoid counter-trend in strong moves) ===
        price_above_sma_50 = close[i] > sma_50[i]
        price_below_sma_50 = close[i] < sma_50[i]
        
        # === ENTRY LOGIC (simplified to ensure trades) ===
        new_signal = 0.0
        
        # --- LONG ENTRY: KAMA bullish + Fisher reversal + vol expansion OR HTF confirmation ---
        if price_above_kama_4h and kama_slope_up:
            # Primary: Fisher reversal + vol expansion
            if (fisher_long or fisher_cross_up) and vol_expanding:
                # HTF confirmation (at least one bullish)
                if price_above_kama_1d or price_above_kama_1w:
                    new_signal = POSITION_SIZE
            # Secondary: Strong HTF bias (allows entry without vol expansion)
            elif price_above_kama_1d and price_above_kama_1w:
                if fisher[i] > fisher_signal[i]:  # Fisher momentum up
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: KAMA bearish + Fisher reversal + vol expansion OR HTF confirmation ---
        elif price_below_kama_4h and kama_slope_down:
            # Primary: Fisher reversal + vol expansion
            if (fisher_short or fisher_cross_down) and vol_expanding:
                # HTF confirmation (at least one bearish)
                if price_below_kama_1d or price_below_kama_1w:
                    new_signal = -POSITION_SIZE
            # Secondary: Strong HTF bias (allows entry without vol expansion)
            elif price_below_kama_1d and price_below_kama_1w:
                if fisher[i] < fisher_signal[i]:  # Fisher momentum down
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if Fisher not at opposite extreme
            if position_side > 0 and fisher[i] < 2.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and fisher[i] > -2.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_kama_4h and price_below_kama_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_kama_4h and price_above_kama_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals