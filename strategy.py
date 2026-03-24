#!/usr/bin/env python3
"""
Experiment #724: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX Regime + RSI Pullback

Hypothesis: 12h timeframe with 1d HTF bias provides optimal balance between trade frequency
(30-50/year) and signal quality. KAMA adapts to market noise better than EMA/HMA.
ADX confirms trend strength, RSI provides pullback entry timing.

Key innovations:
1. KAMA(14) - Kaufman Adaptive Moving Average adapts to volatility/noise
2. 1d HMA(21) for HTF trend bias
3. ADX(14) for trend strength confirmation (>18 = trending, <18 = range)
4. RSI(14) for pullback entries in trend direction
5. ATR(14) 2.5x trailing stoploss
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure trades):
- LONG: 1d HMA bull + KAMA bull + (ADX>18 OR RSI<50 pullback)
- SHORT: 1d HMA bear + KAMA bear + (ADX>18 OR RSI>50 pullback)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_rsi_hma_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts smoothing based on market efficiency
    Formula from "Trading Systems and Methods" by Perry Kaufman
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio (ER) - measures trend vs noise
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA with SMA
    kama[period] = np.mean(close[:period + 1])
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        diff_up = high[i] - high[i-1]
        diff_down = low[i-1] - low[i]
        
        if diff_up > diff_down and diff_up > 0:
            plus_dm[i] = diff_up
        if diff_down > diff_up and diff_down > 0:
            minus_dm[i] = diff_down
    
    # Smoothed DM and TR
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    atr = np.zeros(n)
    
    plus_di[:] = np.nan
    minus_di[:] = np.nan
    atr[:] = np.nan
    
    # First period sum
    sum_plus_dm = np.sum(plus_dm[1:period+1])
    sum_minus_dm = np.sum(minus_dm[1:period+1])
    sum_tr = np.sum(tr[:period])
    
    for i in range(period, n):
        if i == period:
            smoothed_plus_dm = sum_plus_dm
            smoothed_minus_dm = sum_minus_dm
            smoothed_tr = sum_tr
        else:
            smoothed_plus_dm = smoothed_plus_dm - smoothed_plus_dm / period + plus_dm[i]
            smoothed_minus_dm = smoothed_minus_dm - smoothed_minus_dm / period + minus_dm[i]
            smoothed_tr = smoothed_tr - smoothed_tr / period + tr[i]
        
        if smoothed_tr > 1e-10:
            plus_di[i] = 100.0 * smoothed_plus_dm / smoothed_tr
            minus_di[i] = 100.0 * smoothed_minus_dm / smoothed_tr
            atr[i] = smoothed_tr / period
        
        if smoothed_plus_dm + smoothed_minus_dm > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX = smoothed DX
    adx = np.zeros(n)
    adx[:] = np.nan
    
    dx_smooth = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = dx_smooth
    
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    kama = calculate_kama(close, period=14, fast_period=2, slow_period=30)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Also calculate 12h HMA for additional trend confirmation
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
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
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === LOCAL TREND (KAMA + HMA 12h) ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx[i] > 18.0  # Loose threshold for more trades
        trend_weak = adx[i] < 18.0
        
        # === RSI PULLBACK (LOOSE for more trades) ===
        rsi_pullback_long = rsi[i] < 50.0  # Was 40, now 50 for more trades
        rsi_pullback_short = rsi[i] > 50.0  # Was 60, now 50 for more trades
        
        rsi_oversold = rsi[i] < 35.0  # Strong oversold
        rsi_overbought = rsi[i] > 65.0  # Strong overbought
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + KAMA bull + (trend strong OR RSI pullback)
        if htf_1d_bull and kama_bull:
            if trend_strong and hma_bull:
                desired_signal = SIZE_STRONG
            elif rsi_pullback_long:
                desired_signal = SIZE_BASE
            elif rsi_oversold:
                desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + KAMA bear + (trend strong OR RSI pullback)
        elif htf_1d_bear and kama_bear:
            if trend_strong and hma_bear:
                desired_signal = -SIZE_STRONG
            elif rsi_pullback_short:
                desired_signal = -SIZE_BASE
            elif rsi_overbought:
                desired_signal = -SIZE_BASE
        
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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