#!/usr/bin/env python3
"""
Experiment #146: 12h Primary + 1d HTF — KAMA Trend + Donchian Breakout + Volume + Simple Regime

Hypothesis: After analyzing #142 (Sharpe=-0.150), the Choppiness Index regime detection
was too noisy and blocked valid trades. Key improvements:
1. Replace CHOP with KAMA slope for regime (smoother, adapts to volatility)
2. Use KAMA(21) instead of HMA for primary trend (KAMA flattens in chop, trends in trends)
3. Add VOLUME confirmation on breakouts (real breakouts have 1.5x+ avg volume)
4. LOOSENER RSI filters (15-85 range) to ensure trades generate on ALL symbols
5. Simplify stoploss to fixed 3x ATR (trailing was too tight, caused premature exits)
6. Asymmetric sizing: 0.30 with trend, 0.20 counter-trend (reduce counter-trend risk)

Why this should work:
- KAMA's adaptive nature handles BTC/ETH bear markets better than HMA
- Volume filter reduces false breakouts (major issue in #142)
- Looser RSI ensures >=30 trades on train, >=3 on test for ALL symbols
- 1d KAMA provides major trend bias without being too restrictive
- 12h timeframe targets 20-50 trades/year (low fee drag)

Target: Sharpe>0.351, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_donchian_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman's Adaptive Moving Average (KAMA)
    Adapts to market noise - smooth in chop, responsive in trends
    ER (Efficiency Ratio) determines smoothing constant
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(slow_period, n):
        signal = abs(close[i] - close[i - slow_period])
        noise = np.sum(np.abs(np.diff(close[i-slow_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for major trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=50)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30  # 30% with trend
    SIZE_COUNTER = 0.20  # 20% counter-trend (reduced risk)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_12h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d KAMA) ===
        htf_bull = close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_1d_aligned[i]
        
        # === REGIME DETECTION (KAMA slope) ===
        # KAMA rising = trend regime, KAMA flat = chop regime
        kama_slope = 0.0
        if i >= 5 and not np.isnan(kama_12h[i-5]):
            kama_slope = (kama_12h[i] - kama_12h[i-5]) / (kama_12h[i-5] + 1e-10)
        
        is_trending = kama_slope > 0.005  # KAMA rising
        is_choppy = abs(kama_slope) <= 0.005  # KAMA flat
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_bull = close[i] > donchian_upper[i-1]
        donchian_breakout_bear = close[i] < donchian_lower[i-1]
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.5 * vol_sma[i]
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 15.0  # very loose
        rsi_ok_short = rsi[i] < 85.0  # very loose
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === 12h KAMA TREND ===
        kama_bull = close[i] > kama_12h[i]
        kama_bear = close[i] < kama_12h[i]
        
        # === DESIRED SIGNAL (Simplified Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow Donchian breakouts with volume
            # LONG: breakout + volume + RSI ok
            if donchian_breakout_bull and vol_confirmed and rsi_ok_long:
                if htf_bull and kama_bull:
                    desired_signal = SIZE_TREND  # Full size with HTF confirmation
                elif kama_bull:
                    desired_signal = SIZE_TREND * 0.8  # Slightly reduced
            # SHORT: breakout + volume + RSI ok
            elif donchian_breakout_bear and vol_confirmed and rsi_ok_short:
                if htf_bear and kama_bear:
                    desired_signal = -SIZE_TREND  # Full size with HTF confirmation
                elif kama_bear:
                    desired_signal = -SIZE_TREND * 0.8  # Slightly reduced
        else:
            # CHOPPY REGIME: Mean revert at Donchian bounds (counter-trend)
            # LONG: near Donchian lower + RSI oversold
            channel_range = donchian_upper[i] - donchian_lower[i] + 1e-10
            position_in_channel = (close[i] - donchian_lower[i]) / channel_range
            
            if position_in_channel < 0.15 and rsi_oversold:
                desired_signal = SIZE_COUNTER  # Counter-trend long
            elif position_in_channel > 0.85 and rsi_overbought:
                desired_signal = -SIZE_COUNTER  # Counter-trend short
            # Fallback: extreme RSI mean reversion
            elif rsi[i] < 20.0:
                desired_signal = SIZE_COUNTER * 0.7
            elif rsi[i] > 80.0:
                desired_signal = -SIZE_COUNTER * 0.7
        
        # === STOPLOSS CHECK (Fixed 3x ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_TREND * 0.85:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.85:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_COUNTER * 0.85:
            final_signal = SIZE_COUNTER
        elif desired_signal <= -SIZE_COUNTER * 0.85:
            final_signal = -SIZE_COUNTER
        elif desired_signal >= SIZE_COUNTER * 0.5:
            final_signal = SIZE_COUNTER * 0.5
        elif desired_signal <= -SIZE_COUNTER * 0.5:
            final_signal = -SIZE_COUNTER * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss: 3x ATR from entry
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals