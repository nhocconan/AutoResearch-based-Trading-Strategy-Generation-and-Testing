#!/usr/bin/env python3
"""
Experiment #292: 12h Primary + 1d HTF — KAMA Adaptive Trend + RSI Pullback v1

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts better to changing volatility
than HMA/EMA, reducing whipsaws in 2022 crash and 2025 bear market. Combined with
RSI pullback entries (not extremes) and 1d trend filter for direction bias.

Key differences from #284:
1. KAMA instead of HMA - adapts ER (Efficiency Ratio) to volatility
2. SIMPLIFIED entries - RSI 40-60 pullback zone (not extreme CRSI)
3. ADX > 18 filter for trend strength (not too restrictive)
4. Removed funding rate (symbol loading issues caused problems)
5. Looser entry conditions to ensure 30-50 trades/year
6. Asymmetric sizing: 0.30 when 1d aligned, 0.20 otherwise

Entry Logic:
- Long: price > 1d KAMA + 12h KAMA bullish + RSI 40-55 + ADX > 18
- Short: price < 1d KAMA + 12h KAMA bearish + RSI 45-60 + ADX > 18
- Exit: RSI crosses opposite zone OR stoploss hit (2.5x ATR)

Position sizing: 0.20 base, 0.30 when 1d KMA aligned with 12h signal
Stoploss: 2.5x ATR from entry price

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_rsi_adx_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts to market noise via Efficiency Ratio
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i-period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    return kama

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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = strong trend, ADX < 20 = weak/range
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
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
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_12h_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_12h_slow = calculate_kama(close, period=21, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking
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
        
        if np.isnan(kama_12h_fast[i]) or np.isnan(kama_12h_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d KAMA TREND BIAS ===
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === 12h KAMA CROSSOVER ===
        kama_bull = kama_12h_fast[i] > kama_12h_slow[i]
        kama_bear = kama_12h_fast[i] < kama_12h_slow[i]
        
        # === KAMA SLOPE (momentum confirmation) ===
        kama_slope_bull = False
        kama_slope_bear = False
        if i > 5 and not np.isnan(kama_12h_fast[i-5]):
            kama_slope_bull = kama_12h_fast[i] > kama_12h_fast[i-5]
            kama_slope_bear = kama_12h_fast[i] < kama_12h_fast[i-5]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 18.0  # Lowered from 25 for more trades
        
        # === RSI PULLBACK ZONE (not extremes) ===
        rsi_pullback_long = 38.0 <= rsi[i] <= 55.0  # Pullback in uptrend
        rsi_pullback_short = 45.0 <= rsi[i] <= 62.0  # Rally in downtrend
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 1d bull + 12h KAMA bull + RSI pullback + ADX confirmation
        if htf_bull and kama_bull and kama_slope_bull and rsi_pullback_long:
            if adx_strong or above_sma200:  # Either trend strength OR above SMA200
                desired_signal = SIZE_STRONG if adx_strong else SIZE_BASE
        
        # SHORT: 1d bear + 12h KAMA bear + RSI rally + ADX confirmation
        elif htf_bear and kama_bear and kama_slope_bear and rsi_pullback_short:
            if adx_strong or below_sma200:  # Either trend strength OR below SMA200
                desired_signal = -SIZE_STRONG if adx_strong else -SIZE_BASE
        
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
        
        # === EXIT SIGNAL (RSI cross opposite zone) ===
        if in_position and position_side > 0 and rsi[i] > 65.0:
            # Long exit on overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi[i] < 35.0:
            # Short exit on oversold
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