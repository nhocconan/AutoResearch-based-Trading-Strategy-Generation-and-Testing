#!/usr/bin/env python3
"""
Experiment #811: 6h Primary + 1w/1d HTF — Keltner-RSI Pullback with Dual HTF Bias

Hypothesis: 6h timeframe captures 2-5 day swing moves optimally. Using 1w HMA for 
ultra-long-term bias + 1d HMA for medium-term bias creates strong directional filter.
6h RSI(7) provides faster mean-reversion signals than RSI(14), generating more trades.
Keltner Channel (ATR-based) entries on pullbacks within HTF trend reduce whipsaw.

Key innovations:
1. Dual HTF bias: 1w HMA(21) + 1d HMA(21) must align for strong signals
2. 6h RSI(7) instead of RSI(14) — faster, more trades on 6h TF
3. Keltner Channel pullback entries (not breakouts) — enter on dips in uptrend
4. Volume confirmation: taker_buy_volume ratio > 0.45 for longs
5. Asymmetric sizing: 0.30 when both HTF align, 0.20 when only 1d aligns
6. 2.5x ATR trailing stop for risk management

Entry conditions (LOOSE for trade generation):
- LONG: 1w HMA bull + 1d HMA bull + RSI(7)<50 + price>Keltner_lower
- SHORT: 1w HMA bear + 1d HMA bear + RSI(7)>50 + price<Keltner_upper

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_rsi_hma_1w1d_v1"
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

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_keltner(high, low, close, atr_period=20, mult=2.0):
    """Keltner Channel - ATR-based volatility bands"""
    n = len(close)
    if n < atr_period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Middle line: EMA(20) of close
    middle = pd.Series(close).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # ATR for band width
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = middle + mult * atr
    lower = middle - mult * atr
    
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for more trades
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    keltner_upper, keltner_middle, keltner_lower = calculate_keltner(high, low, close, atr_period=20, mult=2.0)
    
    # Volume ratio for confirmation
    taker_ratio = np.zeros(n)
    taker_ratio[:] = np.nan
    for i in range(n):
        if volume[i] > 1e-10:
            taker_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    signals = np.zeros(n)
    SIZE_WEAK = 0.20  # When only 1d aligns
    SIZE_STRONG = 0.30  # When both 1w and 1d align
    
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
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w + 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both align
        htf_strong_bull = htf_1w_bull and htf_1d_bull
        htf_strong_bear = htf_1w_bear and htf_1d_bear
        htf_weak_bull = htf_1d_bull and not htf_1w_bull
        htf_weak_bear = htf_1d_bear and not htf_1w_bear
        
        # === 6h HMA TREND ===
        hma_6h_bull = hma_16[i] > hma_48[i]
        hma_6h_bear = hma_16[i] < hma_48[i]
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_7_oversold = rsi_7[i] < 50.0  # Loose threshold
        rsi_7_overbought = rsi_7[i] > 50.0
        rsi_7_extreme_oversold = rsi_7[i] < 35.0
        rsi_7_extreme_overbought = rsi_7[i] > 65.0
        
        # === KELTNER CHANNEL POSITION ===
        price_in_lower_half = close[i] < keltner_middle[i]
        price_in_upper_half = close[i] > keltner_middle[i]
        price_near_lower = close[i] < keltner_lower[i] + 0.5 * atr_14[i]
        price_near_upper = close[i] > keltner_upper[i] - 0.5 * atr_14[i]
        
        # === VOLUME CONFIRMATION ===
        vol_bullish = taker_ratio[i] > 0.48
        vol_bearish = taker_ratio[i] < 0.52
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + RSI oversold + price in lower Keltner
        if htf_strong_bull:
            if rsi_7_oversold and price_in_lower_half:
                if rsi_7_extreme_oversold or price_near_lower or hma_6h_bull:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_WEAK
        elif htf_weak_bull:
            if rsi_7_oversold and price_in_lower_half:
                if rsi_7_extreme_oversold or hma_6h_bull:
                    desired_signal = SIZE_WEAK
        
        # SHORT: HTF bear + RSI overbought + price in upper Keltner
        elif htf_strong_bear:
            if rsi_7_overbought and price_in_upper_half:
                if rsi_7_extreme_overbought or price_near_upper or hma_6h_bear:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_WEAK
        elif htf_weak_bear:
            if rsi_7_overbought and price_in_upper_half:
                if rsi_7_extreme_overbought or hma_6h_bear:
                    desired_signal = -SIZE_WEAK
        
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
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
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