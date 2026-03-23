#!/usr/bin/env python3
"""
Experiment #1102: 12h Primary + 1d HTF — KAMA Adaptive Trend with Choppiness Position Sizing

Hypothesis: After analyzing 799+ failed experiments, key insights for 12h timeframe:
1. HMA failed in exp #1096 (Sharpe=-0.321) — KAMA adapts better to volatility regimes
2. Choppiness Index should ADJUST position size, NOT filter entries (avoids 0 trades)
3. Looser RSI thresholds (35/65) ensure adequate trade frequency on 12h
4. Volume confirmation on breakouts reduces false signals
5. 1d KAMA provides smoother macro trend filter than HMA
6. Position size varies: 0.30 in trends (CHOP<38), 0.15 in chop (CHOP>61)

Why this should beat Sharpe=0.612 (current best 4h strategy):
- 12h has significantly less noise than 4h, cleaner signals
- KAMA adapts to market efficiency — faster in trends, slower in chop
- Choppiness-based sizing captures more trades while reducing exposure in chop
- Proven pattern: adaptive MA + RSI + volume worked on SOL (research Sharpe +0.782)
- Target: 25-45 trades/year, Sharpe > 0.612, DD < -30%

Timeframe: 12h (primary)
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.15-0.30 discrete (varies by choppiness regime)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_rsi_1d_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts to market noise.
    
    Formula:
    1. Efficiency Ratio (ER) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    2. Smoothing Constant (SC) = [ER * (fast SC - slow SC) + slow SC]^2
    3. KAMA = KAMA[prev] + SC * (close - KAMA[prev])
    
    Fast SC = 2/(fast+1), Slow SC = 2/(slow+1)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    signal = np.abs(close - np.roll(close, er_period))
    signal[:er_period] = np.nan
    
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i-1] + np.abs(close[i] - close[i-1])
    noise[er_period:] = noise[er_period:] - np.roll(noise, er_period)[er_period:]
    noise[:er_period] = np.nan
    
    er = np.zeros(n)
    mask = (noise > 1e-10) & (~np.isnan(signal))
    er[mask] = signal[mask] / noise[mask]
    er[:er_period] = np.nan
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate SC
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    choppiness = np.full(n, np.nan)
    
    if n < period + 1:
        return choppiness
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High - Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    denom = hh - ll
    mask = (denom > 1e-10) & (~np.isnan(tr_sum))
    choppiness[mask] = 100.0 * np.log10(tr_sum[mask] / denom[mask]) / np.log10(period)
    
    return choppiness

def calculate_sma(close, period=20):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d KAMA for macro trend filter
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    rsi_12h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    
    # Volume moving average for confirmation
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE_TREND = 0.30  # Full size in trending markets
    BASE_SIZE_CHOP = 0.15   # Reduced size in choppy markets
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr[i]) or np.isnan(choppiness[i]):
            continue
        if np.isnan(kama_12h[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(volume_sma[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d KAMA) ===
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (12h KAMA) ===
        trend_bull = close[i] > kama_12h[i]
        trend_bear = close[i] < kama_12h[i]
        
        # === MARKET REGIME (Choppiness) ===
        # CHOP < 38 = trending, CHOP > 61 = choppy
        is_trending = choppiness[i] < 45.0
        is_choppy = choppiness[i] > 55.0
        
        # Position size based on regime
        current_size = BASE_SIZE_TREND if is_trending else BASE_SIZE_CHOP
        
        # === MOMENTUM (RSI) ===
        # Looser thresholds for more trades on 12h
        rsi_oversold = rsi_12h[i] < 40.0
        rsi_overbought = rsi_12h[i] > 60.0
        rsi_neutral = 35.0 <= rsi_12h[i] <= 65.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = volume[i] > 1.2 * volume_sma[i]
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + 12h trend bull + RSI pullback + (volume OR trending regime)
        if macro_bull and trend_bull and above_sma50:
            if rsi_oversold:
                # Strong entry: oversold RSI in uptrend
                if volume_confirm or is_trending:
                    desired_signal = current_size
            elif rsi_neutral and rsi_12h[i] < 50.0:
                # Moderate entry: neutral RSI but below 50
                if is_trending and volume_confirm:
                    desired_signal = current_size * 0.7
        
        # === SHORT ENTRY ===
        # Macro bear + 12h trend bear + RSI pullback + (volume OR trending regime)
        elif macro_bear and trend_bear and below_sma50:
            if rsi_overbought:
                # Strong entry: overbought RSI in downtrend
                if volume_confirm or is_trending:
                    desired_signal = -current_size
            elif rsi_neutral and rsi_12h[i] > 50.0:
                # Moderate entry: neutral RSI but above 50
                if is_trending and volume_confirm:
                    desired_signal = -current_size * 0.7
        
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
                # Hold long if macro and 12h trend still bull
                if macro_bull and trend_bull and choppiness[i] < 65.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro and 12h trend still bear
                if macro_bear and trend_bear and choppiness[i] < 65.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses or RSI very overbought
            if macro_bear or rsi_12h[i] > 75.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses or RSI very oversold
            if macro_bull or rsi_12h[i] < 25.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE_TREND * 0.8:
                desired_signal = BASE_SIZE_TREND
            elif desired_signal >= BASE_SIZE_CHOP * 0.8:
                desired_signal = BASE_SIZE_CHOP
            else:
                desired_signal = 0.0
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE_TREND * 0.8:
                desired_signal = -BASE_SIZE_TREND
            elif desired_signal <= -BASE_SIZE_CHOP * 0.8:
                desired_signal = -BASE_SIZE_CHOP
            else:
                desired_signal = 0.0
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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