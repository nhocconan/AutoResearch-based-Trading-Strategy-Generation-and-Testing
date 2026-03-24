#!/usr/bin/env python3
"""
Experiment #727: 6h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX/RSI Confluence

Hypothesis: 6h timeframe offers optimal balance between noise reduction and trade frequency.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - smooth during trends,
sensitive during ranges. Combined with ADX for trend strength and RSI for entry timing.

Key innovations:
1. KAMA(10) - adapts smoothing based on market efficiency ratio
2. ADX(14) > 20 for trend confirmation (looser than typical 25 threshold)
3. RSI(14) with 35/65 thresholds for entry timing (not extreme 20/80)
4. Volume ratio confirmation (current vol > 1.2x 20-period avg)
5. 1d KAMA for HTF bias, 1w HMA for macro trend direction
6. ATR(14) 2.5x trailing stop with signal→0 on stoploss
7. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry logic (LOOSE to ensure trades):
- LONG: 1w HMA bull + 1d KAMA bull + ADX>20 + RSI>35 + volume confirm + price>6h KAMA
- SHORT: 1w HMA bear + 1d KAMA bear + ADX>20 + RSI<65 + volume confirm + price<6h KAMA
- Mean revert fallback: RSI<30 or RSI>70 with HTF bias (ensures trades in range markets)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_adx_rsi_volume_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts smoothing based on market efficiency
    ER (Efficiency Ratio) determines how much to smooth
    High ER (trending) = less smoothing, Low ER (ranging) = more smoothing
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(slow_period, n):
        change = abs(close[i] - close[i - slow_period])
        if change < 1e-10:
            er[i] = 0.0
            continue
        volatility = 0.0
        for j in range(i - slow_period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        if volatility < 1e-10:
            er[i] = 0.0
        else:
            er[i] = change / volatility
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, plus_di, minus_di

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
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    kama_6h = calculate_kama(close, period=10)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (current vs 20-period average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    
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
        
        if np.isnan(kama_6h[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA + 1d KAMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        htf_1d_bull = close[i] > kama_1d_aligned[i]
        htf_1d_bear = close[i] < kama_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx[i] > 20.0  # Looser than typical 25
        trend_very_strong = adx[i] > 30.0
        
        # === MOMENTUM (RSI) ===
        rsi_bull = rsi[i] > 35.0  # Looser than 50
        rsi_bear = rsi[i] < 65.0  # Looser than 50
        rsi_oversold = rsi[i] < 30.0
        rsi_overbought = rsi[i] > 70.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = vol_ratio[i] > 1.2
        
        # === KAMA TREND (6h) ===
        kama_bull = close[i] > kama_6h[i]
        kama_bear = close[i] < kama_6h[i]
        
        # === DI DIRECTION ===
        di_bull = plus_di[i] > minus_di[i]
        di_bear = minus_di[i] > plus_di[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: Strong confluence
        long_strong = (htf_1w_bull and htf_1d_bull and kama_bull and 
                       (trend_strong or rsi_bull) and volume_confirm)
        # LONG: Moderate confluence (ensures trades in weaker trends)
        long_moderate = (htf_1w_bull and kama_bull and rsi_bull and rsi[i] > 45.0)
        # LONG: Mean revert in bull market
        long_mr = (htf_1w_bull and rsi_oversold)
        
        # SHORT: Strong confluence
        short_strong = (htf_1w_bear and htf_1d_bear and kama_bear and 
                        (trend_strong or rsi_bear) and volume_confirm)
        # SHORT: Moderate confluence
        short_moderate = (htf_1w_bear and kama_bear and rsi_bear and rsi[i] < 55.0)
        # SHORT: Mean revert in bear market
        short_mr = (htf_1w_bear and rsi_overbought)
        
        if long_strong:
            desired_signal = SIZE_STRONG
        elif long_moderate:
            desired_signal = SIZE_BASE
        elif long_mr:
            desired_signal = SIZE_BASE
        elif short_strong:
            desired_signal = -SIZE_STRONG
        elif short_moderate:
            desired_signal = -SIZE_BASE
        elif short_mr:
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