#!/usr/bin/env python3
"""
Experiment #760: 6h Primary + 1d/1w HTF — Fisher Transform Reversals with KAMA Trend

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Fisher Transform
excels at catching reversals in bear/range markets (BTC 2022 crash, 2025 bear).
Combined with KAMA (adaptive, less whipsaw than HMA) and 1w HTF bias for major trend.

Key innovations:
1. 1w HMA(21) for major trend bias — slow, reliable, avoids whipsaw
2. 1d Fisher Transform(9) for reversal entries — proven in bear markets
3. 6h KAMA(10) for adaptive trend confirmation — adjusts to market efficiency
4. 6h Keltner Channel squeeze — volatility expansion breakouts
5. Volume confirmation via taker_buy_volume ratio
6. Asymmetric sizing: stronger signals in HTF trend direction

Entry logic:
- LONG: 1w HMA bull + Fisher < -1.5 (oversold reversal) + KAMA bull + KC squeeze breakout
- SHORT: 1w HMA bear + Fisher > +1.5 (overbought reversal) + KAMA bear + KC squeeze breakout

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete (max 0.35)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_kama_kc_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i-slow):i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_fisher(close, period=9):
    """Ehlers Fisher Transform - normalizes price to Gaussian distribution"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.max(close[i-period+1:i+1])
        ll = np.min(close[i-period+1:i+1])
        
        if hh > ll:
            # Normalize price to -1 to +1 range
            value = 0.66 * ((close[i] - ll) / (hh - ll) - 0.5) + 0.67 * (fisher[i-1] if not np.isnan(fisher[i-1]) else 0)
            value = np.clip(value, -0.999, 0.999)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
            trigger[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else fisher[i]
        else:
            fisher[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else 0
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_keltner(close, high, low, period=20, atr_mult=2.0):
    """Keltner Channels - volatility-based bands"""
    n = len(close)
    if n < period + 14:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # EMA for middle line
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # ATR for channel width
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    return ema, upper, lower

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(taker_buy_volume, volume):
    """Taker buy volume ratio - institutional flow indicator"""
    n = len(volume)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(n):
        if volume[i] > 1e-10:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    fisher_1d_raw, trigger_1d_raw = calculate_fisher(df_1d['close'].values, period=9)
    fisher_1d_aligned = align_htf_to_ltf(prices, df_1d, fisher_1d_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, period=10, fast=2, slow=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    kc_mid, kc_upper, kc_lower = calculate_keltner(close, high, low, period=20, atr_mult=2.0)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # Rolling volume ratio for confirmation
    vol_ratio_ma = pd.Series(vol_ratio).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_MAX = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_6h[i]) or np.isnan(kc_mid[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(fisher_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d Fisher Transform (reversal signal) ===
        fisher_1d = fisher_1d_aligned[i]
        fisher_oversold = fisher_1d < -1.5
        fisher_overbought = fisher_1d > 1.5
        
        # Fisher crossover for entry timing
        fisher_cross_up = False
        fisher_cross_down = False
        if i > 0 and not np.isnan(fisher_1d_aligned[i-1]):
            fisher_cross_up = (fisher_1d_aligned[i-1] < -1.5) and (fisher_1d >= -1.5)
            fisher_cross_down = (fisher_1d_aligned[i-1] > 1.5) and (fisher_1d <= 1.5)
        
        # === 6h KAMA Trend ===
        kama_bull = close[i] > kama_6h[i]
        kama_bear = close[i] < kama_6h[i]
        
        # KAMA slope
        kama_slope_bull = False
        kama_slope_bear = False
        if i > 5 and not np.isnan(kama_6h[i-5]):
            kama_slope_bull = kama_6h[i] > kama_6h[i-5]
            kama_slope_bear = kama_6h[i] < kama_6h[i-5]
        
        # === Keltner Channel Squeeze/Breakout ===
        kc_squeeze = (kc_upper[i] - kc_lower[i]) < (kc_upper[i-20] - kc_lower[i-20]) * 0.8 if i > 20 and not np.isnan(kc_upper[i-20]) else False
        kc_breakout_long = close[i] > kc_upper[i]
        kc_breakout_short = close[i] < kc_lower[i]
        
        # === Volume Confirmation ===
        vol_confirmed_long = vol_ratio_ma[i] > 0.55 if not np.isnan(vol_ratio_ma[i]) else False
        vol_confirmed_short = vol_ratio_ma[i] < 0.45 if not np.isnan(vol_ratio_ma[i]) else False
        
        # === ENTRY LOGIC (Fisher reversals with HTF bias) ===
        desired_signal = 0.0
        signal_strength = 0
        
        # LONG: 1w bull + Fisher oversold/reversal + KAMA confirm + volume
        if htf_1w_bull:
            if fisher_oversold or fisher_cross_up:
                if kama_bull and kama_slope_bull:
                    signal_strength = 2
                    if vol_confirmed_long or kc_breakout_long:
                        signal_strength = 3
                else:
                    signal_strength = 1
        
        # SHORT: 1w bear + Fisher overbought/reversal + KAMA confirm + volume
        elif htf_1w_bear:
            if fisher_overbought or fisher_cross_down:
                if kama_bear and kama_slope_bear:
                    signal_strength = 2
                    if vol_confirmed_short or kc_breakout_short:
                        signal_strength = 3
                else:
                    signal_strength = 1
        
        # Set signal based on strength
        if signal_strength >= 3:
            desired_signal = SIZE_STRONG if htf_1w_bull else -SIZE_STRONG
        elif signal_strength == 2:
            desired_signal = SIZE_BASE if htf_1w_bull else -SIZE_BASE
        elif signal_strength == 1:
            # Weak signal - only enter if already in position
            if not in_position:
                desired_signal = 0.0
            else:
                desired_signal = SIZE_BASE * 0.5 if position_side > 0 else -SIZE_BASE * 0.5
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
        elif desired_signal > 0:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal < 0:
            final_signal = -SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals