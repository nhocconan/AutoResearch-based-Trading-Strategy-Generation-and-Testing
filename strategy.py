#!/usr/bin/env python3
"""
Experiment #019: 1h Primary + 4h/12h HTF — Volatility Spike Reversion + HTF Trend + RSI Pullback

Hypothesis: After 18 failed experiments, the pattern shows:
- Session filters cause 0 trades (#009, #013, #016) — AVOID session filters
- cRSI + Choppiness combinations failed repeatedly — try different indicators
- Fisher Transform failed (#008, #015) — avoid complex transforms
- Volume-based strategies failed — skip volume filters
- SUCCESS pattern: HTF HMA + LTF RSI pullback (from current best mtf_hma_rsi_zscore_v1)

NEW APPROACH for 1h:
- Volatility Spike Reversion: ATR(7)/ATR(30) > 1.8 signals panic/reversion opportunity
- 4h HMA(21) for major trend bias (proven edge)
- 12h HMA(50) for secondary trend confirmation
- 1h RSI(14) for pullback entries (loose: 35/65 thresholds to ensure trades)
- 1h Bollinger(20,2) for mean reversion bounds
- Asymmetric sizing: 0.25 normal, 0.35 on high conviction (vol spike + HTF aligned)
- Stoploss: 2.5x ATR trailing

Why this might work:
- Vol spikes precede reversals (panic selling/buying exhaustion)
- HTF trend alignment reduces false signals in bear markets
- Loose RSI ensures >=30 trades on train (critical requirement)
- No session filter (learned from failures)
- 1h TF with HTF bias = ~50 trades/year target

Target: Sharpe>0.019 (beat current best), DD>-40%, trades>=30 train, >=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_volspike_hma_rsi_bb_4h12h_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for secondary trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Volatility spike ratio: ATR(7)/ATR(30)
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            vol_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE_NORMAL = 0.25  # 25% position size
    SIZE_HIGH = 0.35    # 35% on high conviction
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h + 12h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong HTF alignment (both 4h and 12h agree)
        htf_strong_bull = htf_4h_bull and htf_12h_bull
        htf_strong_bear = htf_4h_bear and htf_12h_bear
        
        # === VOLATILITY REGIME ===
        # Vol spike: ratio > 1.8 signals panic/reversion opportunity
        vol_spike = vol_ratio[i] > 1.8
        vol_normal = vol_ratio[i] <= 1.8
        
        # === 1h HMA TREND ===
        hma_1h_bull = close[i] > hma_1h[i]
        hma_1h_bear = close[i] < hma_1h[i]
        
        # === RSI PULLBACK (LOOSE thresholds to ensure trades) ===
        rsi_oversold = rsi[i] < 45.0  # loose for entries
        rsi_overbought = rsi[i] > 55.0  # loose for entries
        rsi_extreme_low = rsi[i] < 35.0
        rsi_extreme_high = rsi[i] > 65.0
        
        # === BOLLINGER POSITION ===
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range > 1e-10:
            bb_position = (close[i] - bb_lower[i]) / bb_range
        else:
            bb_position = 0.5
        
        near_bb_lower = bb_position < 0.15
        near_bb_upper = bb_position > 0.85
        near_bb_mid = 0.4 < bb_position < 0.6
        
        # === DESIRED SIGNAL (Multi-confluence logic) ===
        desired_signal = 0.0
        conviction = 1.0  # 1.0 = normal size, 1.4 = high conviction
        
        # LONG SETUP: HTF bull + RSI pullback + vol regime
        if htf_strong_bull:
            # High conviction: vol spike + RSI extreme + near BB lower
            if vol_spike and rsi_extreme_low and near_bb_lower:
                desired_signal = SIZE_HIGH
                conviction = 1.4
            # Normal: RSI pullback + HTF aligned
            elif rsi_oversold and hma_1h_bull:
                desired_signal = SIZE_NORMAL
            # BB mean reversion in uptrend
            elif near_bb_lower and htf_4h_bull:
                desired_signal = SIZE_NORMAL * 0.8
        
        # SHORT SETUP: HTF bear + RSI pullback + vol regime
        elif htf_strong_bear:
            # High conviction: vol spike + RSI extreme + near BB upper
            if vol_spike and rsi_extreme_high and near_bb_upper:
                desired_signal = -SIZE_HIGH
                conviction = 1.4
            # Normal: RSI pullback + HTF aligned
            elif rsi_overbought and hma_1h_bear:
                desired_signal = -SIZE_NORMAL
            # BB mean reversion in downtrend
            elif near_bb_upper and htf_4h_bear:
                desired_signal = -SIZE_NORMAL * 0.8
        
        # NEUTRAL HTF: Use 1h signals only (range market)
        else:
            # Mean reversion at BB extremes
            if near_bb_lower and rsi_extreme_low:
                desired_signal = SIZE_NORMAL * 0.7
            elif near_bb_upper and rsi_extreme_high:
                desired_signal = -SIZE_NORMAL * 0.7
            # Vol spike reversal
            elif vol_spike and rsi_extreme_low:
                desired_signal = SIZE_NORMAL * 0.6
            elif vol_spike and rsi_extreme_high:
                desired_signal = -SIZE_NORMAL * 0.6
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_HIGH * 0.85:
            final_signal = SIZE_HIGH
        elif desired_signal <= -SIZE_HIGH * 0.85:
            final_signal = -SIZE_HIGH
        elif desired_signal >= SIZE_NORMAL * 0.85:
            final_signal = SIZE_NORMAL
        elif desired_signal <= -SIZE_NORMAL * 0.85:
            final_signal = -SIZE_NORMAL
        elif desired_signal >= SIZE_NORMAL * 0.5:
            final_signal = SIZE_NORMAL * 0.5
        elif desired_signal <= -SIZE_NORMAL * 0.5:
            final_signal = -SIZE_NORMAL * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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