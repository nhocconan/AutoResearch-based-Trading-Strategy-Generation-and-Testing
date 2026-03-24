#!/usr/bin/env python3
"""
Experiment #033: 1d Primary + 1w HTF — Dual Regime with Relaxed Entries

Hypothesis: Daily timeframe with weekly trend filter should produce 20-50 trades/year.
Building on #021's framework but with CRITICAL fixes for 0-trade failures:

1. RELAXED entry thresholds - CRSI 20/80 instead of 15/85 (more trades)
2. Donchian breakout as primary signal (proven on SOL in history)
3. Choppiness regime switch but with wider bands (50/60 instead of 45/55)
4. 1w HMA as simple trend bias (not dual 1d+1w which was too restrictive)
5. ATR trailing stop at 3x (wider than 2.5x to avoid premature exits)
6. Size: 0.30 flat (no complex sizing that reduces trade frequency)

Key difference from failed #023/#027: Much looser entry conditions to ensure
minimum 10 trades/symbol on train. Daily TF naturally has fewer signals, so
each condition must be achievable.

Entry Logic:
- CHOPPY (CHOP>60): RSI<25 long, RSI>75 short (mean reversion)
- TREND (CHOP<50): Donchian breakout + 1w HMA confirmation
- NEUTRAL: Trade either direction with reduced size

Risk: 3x ATR trailing stop, signal magnitude 0.30, discrete levels
Target: Sharpe>0.4, trades>20/symbol train, >5/symbol test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_rsi_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_hma(close, period=21):
    """Hull Moving Average - smooth trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection (range vs trend)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

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

def calculate_sma(close, period=200):
    """Simple Moving Average - long-term trend filter"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # Wider bands for more trades: 50-60 neutral zone
        is_choppy = chop[i] > 60.0
        is_trending = chop[i] < 50.0
        
        # === HTF TREND BIAS (1w HMA) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === SMA200 FILTER (long-term bias) ===
        above_sma200 = not np.isnan(sma200[i]) and close[i] > sma200[i]
        below_sma200 = not np.isnan(sma200[i]) and close[i] < sma200[i]
        
        # === DESIRED SIGNAL BASED ON REGIME ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - RSI extremes (RELAXED: 25/75 instead of 20/80)
            # Long: RSI < 25 (oversold in range)
            if rsi[i] < 25.0:
                # Prefer long when above SMA200 or 1w HMA bullish
                if above_sma200 or hma_1w_bull:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = BASE_SIZE * 0.67  # Reduced size against trend
            
            # Short: RSI > 75 (overbought in range)
            elif rsi[i] > 75.0:
                # Prefer short when below SMA200 or 1w HMA bearish
                if below_sma200 or hma_1w_bear:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -BASE_SIZE * 0.67  # Reduced size against trend
        
        elif is_trending:
            # TREND REGIME - Donchian breakout + HTF confirmation
            # Long breakout: price at Donchian upper
            if close[i] >= donchian_upper[i] * 0.999:  # Near upper band
                if hma_1w_bull:  # Weekly trend confirms
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = BASE_SIZE * 0.67  # Reduced without HTF confirm
            
            # Short breakout: price at Donchian lower
            elif close[i] <= donchian_lower[i] * 1.001:  # Near lower band
                if hma_1w_bear:  # Weekly trend confirms
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -BASE_SIZE * 0.67  # Reduced without HTF confirm
        
        else:
            # NEUTRAL REGIME (50 <= CHOP <= 60) - trade breakouts with any HTF bias
            # Long breakout
            if close[i] >= donchian_upper[i] * 0.999:
                desired_signal = BASE_SIZE * 0.67
            # Short breakout
            elif close[i] <= donchian_lower[i] * 1.001:
                desired_signal = -BASE_SIZE * 0.67
            # RSI mean reversion in neutral
            elif rsi[i] < 30.0:
                desired_signal = BASE_SIZE * 0.67
            elif rsi[i] > 70.0:
                desired_signal = -BASE_SIZE * 0.67
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.67
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.67
        elif abs(desired_signal) >= 0.10:
            final_signal = np.sign(desired_signal) * BASE_SIZE * 0.67
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals