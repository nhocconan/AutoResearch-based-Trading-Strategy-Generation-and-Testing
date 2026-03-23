#!/usr/bin/env python3
"""
Experiment #694: 4h Primary + 1d/1w HTF — KAMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to volatility better than 
EMA/HMA, reducing whipsaw in choppy markets. Combined with RSI pullback entries 
(entering on dips in uptrends, rallies in downtrends) and Choppiness Index regime 
filter, this should work in both trending and ranging markets.

Key improvements over failed experiments:
1. KAMA instead of HMA/EMA — adapts ER (Efficiency Ratio) to market conditions
2. Looser RSI thresholds (35/65) to ensure trade frequency (avoid 0 trades issue)
3. Single HTF filter (1d KAMA) instead of requiring 1d+1w agreement (too strict)
4. Choppiness Index only for regime detection, not as entry filter
5. Simple 2.5x ATR trailing stop — proven to work in past experiments
6. Position size 0.28 — balances return vs drawdown

Why 4h works:
- #689 (4h Fisher+KAMA+ADX) had +12.8% return despite negative Sharpe
- #691 (4h CRSI+Donchian+Chop) had +18.4% return
- 4h TF gives 20-50 trades/year target, avoiding fee drag of lower TF

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_chop_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    signal = np.abs(close - np.roll(close, er_period))
    signal[:er_period] = np.nan
    
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i-1] + np.abs(close[i] - close[i-1])
    noise_change = np.abs(noise - np.roll(noise, er_period))
    noise_change[:er_period] = np.nan
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = signal / (noise_change + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_sum = np.sum(tr[i-period+1:i+1])
        
        if highest_high - lowest_low > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_4h = calculate_rsi(close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    REDUCED_SIZE = 0.18
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or np.isnan(kama_4h[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(kama_1d_aligned[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (1d KAMA) ===
        trend_bullish_1d = close[i] > kama_1d_aligned[i]
        trend_bearish_1d = close[i] < kama_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_4h[i] > 55  # Range market
        is_trending = chop_4h[i] < 45  # Trend market
        
        # === 4h KAMA TREND ===
        kama_bullish_4h = close[i] > kama_4h[i]
        kama_bearish_4h = close[i] < kama_4h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # Reduce size in high volatility
        atr_median = np.nanmedian(atr_4h[max(0, i-100):i+1])
        if atr_median > 1e-10 and atr_4h[i] > 1.5 * atr_median:
            current_size = REDUCED_SIZE
        
        # === LONG ENTRIES ===
        # Trending market: pullback to KAMA with RSI dip
        if is_trending and trend_bullish_1d and above_sma200:
            if kama_bullish_4h and rsi_4h[i] < 55 and rsi_4h[i] > 35:
                desired_signal = current_size
        
        # Choppy market: mean reversion at lower range
        elif is_choppy and trend_bullish_1d:
            if rsi_4h[i] < 40 and close[i] < kama_4h[i] * 0.98:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Trending market: rally to KAMA with RSI rise
        if is_trending and trend_bearish_1d and below_sma200:
            if kama_bearish_4h and rsi_4h[i] > 45 and rsi_4h[i] < 65:
                desired_signal = -current_size
        
        # Choppy market: mean reversion at upper range
        elif is_choppy and trend_bearish_1d:
            if rsi_4h[i] > 60 and close[i] > kama_4h[i] * 1.02:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d trend still bullish and RSI not overbought
                if trend_bullish_1d and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend still bearish and RSI not oversold
                if trend_bearish_1d and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: RSI overbought OR 1d trend reverses
        if in_position and position_side > 0:
            if rsi_4h[i] > 75:
                desired_signal = 0.0
            elif close[i] < kama_1d_aligned[i] * 0.97:
                desired_signal = 0.0
        
        # Short exit: RSI oversold OR 1d trend reverses
        if in_position and position_side < 0:
            if rsi_4h[i] < 25:
                desired_signal = 0.0
            elif close[i] > kama_1d_aligned[i] * 1.03:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals