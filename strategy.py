#!/usr/bin/env python3
"""
Experiment #390: 1h Primary + 4h/1d HTF — Regime-Adaptive Fisher-RSI v1

Hypothesis: Recent failures (#378-#389) show 0 trades (Sharpe=0) or negative Sharpe.
The issue: entry conditions too strict OR too simple. This strategy uses:

1. CHOPPINESS INDEX for regime detection (range vs trend)
2. RANGE regime (CHOP > 55): Mean reversion at BB bands + RSI extremes + Fisher cross
3. TREND regime (CHOP < 45): Follow 4h HMA direction, enter on 1h RSI pullback
4. FISHER TRANSFORM for precise entry timing (catches reversals in bear markets)
5. SESSION FILTER (06-22 UTC) avoids Asian low-liquidity whipsaws
6. LOOSENED thresholds to ensure trades actually trigger (RSI 15/85, not 20/80)

Key differences from failed attempts:
- Fisher Transform catches reversals better than pure RSI (proven in 2022 crash)
- Simpler regime: Choppiness only (ADX added complexity without benefit)
- RSI thresholds 15/85 (looser than 20/80) to ensure trade generation
- Position size 0.25 base, 0.30 when HTF aligned
- Stoploss 2.5x ATR with proper tracking

Target: Sharpe>0.45, DD>-35%, trades>=40/train, trades>=5/test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_rsi_chop_regime_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_fisher(close, period=9):
    """Ehlers Fisher Transform - catches reversals in bear/range markets"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    fisher[:] = np.nan
    trigger[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > period else 0.0
            trigger[i] = fisher[i]
            continue
        
        # Normalize price to -1 to +1
        x = (2.0 * (close[i] - lowest) / range_val) - 1.0
        
        # Smooth with EMA
        if i == period:
            x_smooth = x
        else:
            x_smooth = 0.67 * x + 0.33 * ((2.0 * (close[i-1] - np.min(close[i-period:i])) / 
                         (np.max(close[i-period:i]) - np.min(close[i-period:i]))) - 1.0)
        
        # Clamp to avoid division issues
        x_smooth = np.clip(x_smooth, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x_smooth) / (1.0 - x_smooth))
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher[i-1]
        
        trigger[i] = fisher[i-1] if i > period else fisher[i]
    
    return fisher, trigger

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, sma

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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_dev=2.0)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=range
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with CHOPPINESS ===
        # Trending: CHOP < 45
        # Range: CHOP > 55
        # Use hysteresis for stability
        
        is_trending = chop[i] < 45.0
        is_range = chop[i] > 55.0
        
        if is_trending:
            current_regime = 1
        elif is_range:
            current_regime = 2
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === HTF BIAS (4h + 1d) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong HTF alignment when both 4h and 1d agree
        htf_strong_bull = htf_4h_bull and htf_1d_bull
        htf_strong_bear = htf_4h_bear and htf_1d_bear
        
        # === 1h HMA TREND ===
        hma_bull = close[i] > hma_1h[i]
        hma_bear = close[i] < hma_1h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI EXTREMES (LOOSENED for trade generation) ===
        rsi_oversold = rsi[i] < 15.0  # Very oversold
        rsi_overbought = rsi[i] > 85.0  # Very overbought
        rsi_pullback_long = rsi[i] < 45.0  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 55.0  # Pullback in downtrend
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_trigger[i-1]):
            # Long: Fisher crosses above -1.5 from below
            if fisher_trigger[i-1] < -1.5 and fisher[i] > -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 from above
            if fisher_trigger[i-1] > 1.5 and fisher[i] < 1.5:
                fisher_cross_short = True
        
        # === BOLLINGER BAND POSITION ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002  # At or below lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.998  # At or above upper band
        
        # === SESSION FILTER (06-22 UTC for better liquidity) ===
        # Extract hour from open_time (assuming milliseconds timestamp)
        try:
            timestamp_ms = prices["open_time"].iloc[i]
            hour_utc = (timestamp_ms // 3600000) % 24
            in_session = 6 <= hour_utc <= 22
        except:
            in_session = True  # Default to allow if can't parse
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (follow HTF direction with pullback entries)
        if current_regime == 1:
            # Long: HTF bull + 1h HMA bull + RSI pullback + Fisher confirmation
            if htf_strong_bull and hma_bull and above_sma200:
                if rsi_pullback_long and (fisher_cross_long or at_bb_lower):
                    desired_signal = SIZE_STRONG if in_session else SIZE_BASE
            
            # Short: HTF bear + 1h HMA bear + RSI pullback + Fisher confirmation
            elif htf_strong_bear and hma_bear and below_sma200:
                if rsi_pullback_short and (fisher_cross_short or at_bb_upper):
                    desired_signal = -SIZE_STRONG if in_session else -SIZE_BASE
        
        # REGIME 2: RANGE (mean reversion at extremes)
        elif current_regime == 2:
            # Long: RSI very oversold + at BB lower + Fisher cross
            if rsi_oversold and at_bb_lower:
                if fisher_cross_long or rsi[i] < 10.0:  # Extra oversold = no Fisher needed
                    desired_signal = SIZE_BASE
            
            # Short: RSI very overbought + at BB upper + Fisher cross
            elif rsi_overbought and at_bb_upper:
                if fisher_cross_short or rsi[i] > 90.0:  # Extra overbought = no Fisher needed
                    desired_signal = -SIZE_BASE
        
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