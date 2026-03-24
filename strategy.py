#!/usr/bin/env python3
"""
Experiment #283: 6h Primary + 1d/1w HTF — Vol Spike Reversion + Regime Adaptive v1

Hypothesis: 6h timeframe is ideal for capturing volatility mean-reversion after panic spikes.
Unlike 4h (too noisy) or 12h (too slow), 6h catches vol crush patterns with fewer false signals.

Key innovations vs failed 6h experiments:
1. VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 signals panic exhaustion → enter counter-trend
2. REGIME ADAPTIVE: ADX(14) determines trend vs range → different entry logic per regime
3. HTF BIAS: 1d HMA(50) for intermediate trend, 1w HMA(21) for major bias
4. FEWER TRADES: Target 20-50 trades/year by requiring 3+ confluence factors

Regime Detection:
- ADX > 25 = trending → trade pullbacks to EMA(21) with HTF direction
- ADX < 20 = ranging → mean revert at Bollinger Band extremes
- 20-25 = transition (use previous regime memory for hysteresis)

Vol Spike Entry (both regimes):
- Long: ATR_ratio > 2.0 + close < BB_lower + RSI < 35 + 1d HMA bull
- Short: ATR_ratio > 2.0 + close > BB_upper + RSI > 65 + 1d HMA bear

Trend Pullback Entry (trending regime only):
- Long: price > EMA21 + RSI 40-55 + 1w HMA bull
- Short: price < EMA21 + RSI 45-60 + 1w HMA bear

Exit: ATR_ratio < 1.3 (vol normalized) OR stoploss hit (2.5x ATR trailing)

Position sizing: 0.25 base, 0.30 for vol spike signals (higher conviction)
Stoploss: 2.5x ATR trailing from entry

Target: Sharpe>0.45 (beat current best 0.399), DD>-35%, trades>=15 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volspike_regime_adaptive_1d1w_v1"
timeframe = "6h"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 2 + 1:
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
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    # DX
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    ema_21 = calculate_ema(close, 21)
    atr = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # ATR ratio for vol spike detection
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_VOLSPIKE = 0.30  # Higher conviction for vol spike entries
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=ranging
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_vol_spike = False  # Track if entered on vol spike
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        trend_threshold = 25.0
        range_threshold = 20.0
        
        if adx[i] > trend_threshold:
            current_regime = 1  # trending
        elif adx[i] < range_threshold:
            current_regime = 2  # ranging
        else:
            current_regime = prev_regime  # use memory
        
        prev_regime = current_regime
        
        # === HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1w for major trend
        htf_1w_valid = not np.isnan(hma_1w_aligned[i])
        htf_1w_bull = htf_1w_valid and close[i] > hma_1w_aligned[i]
        htf_1w_bear = htf_1w_valid and close[i] < hma_1w_aligned[i]
        
        # === VOL SPIKE DETECTION ===
        vol_spike = False
        vol_normalized = False
        if not np.isnan(atr_ratio[i]):
            vol_spike = atr_ratio[i] > 2.0
            vol_normalized = atr_ratio[i] < 1.3
        
        # === PRICE POSITION ===
        at_bb_lower = close[i] <= bb_lower[i] * 1.002  # Within 0.2% of lower band
        at_bb_upper = close[i] >= bb_upper[i] * 0.998  # Within 0.2% of upper band
        above_ema21 = close[i] > ema_21[i]
        below_ema21 = close[i] < ema_21[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        is_vol_spike_entry = False
        
        # VOL SPIKE REVERSION (both regimes - highest conviction)
        if vol_spike:
            # Long: panic low + oversold RSI + 1d bullish
            if at_bb_lower and rsi[i] < 35 and htf_1d_bull:
                desired_signal = SIZE_VOLSPIKE
                is_vol_spike_entry = True
            
            # Short: panic high + overbought RSI + 1d bearish
            elif at_bb_upper and rsi[i] > 65 and htf_1d_bear:
                desired_signal = -SIZE_VOLSPIKE
                is_vol_spike_entry = True
        
        # REGIME 1: TRENDING (pullback entries)
        elif current_regime == 1:
            # Long pullback: above EMA21 + RSI dipping but not oversold + HTF bull
            if above_ema21 and 40 <= rsi[i] <= 55 and htf_1d_bull:
                # Require 1w confirmation for stronger signals
                if htf_1w_bull:
                    desired_signal = SIZE_BASE
                elif htf_1w_valid:
                    desired_signal = SIZE_BASE * 0.8
                else:
                    desired_signal = SIZE_BASE * 0.6
            
            # Short pullback: below EMA21 + RSI rising but not overbought + HTF bear
            elif below_ema21 and 45 <= rsi[i] <= 60 and htf_1d_bear:
                if htf_1w_bear:
                    desired_signal = -SIZE_BASE
                elif htf_1w_valid:
                    desired_signal = -SIZE_BASE * 0.8
                else:
                    desired_signal = -SIZE_BASE * 0.6
        
        # REGIME 2: RANGING (mean reversion at BB extremes)
        elif current_regime == 2:
            # Long: at BB lower + RSI oversold
            if at_bb_lower and rsi[i] < 35:
                desired_signal = SIZE_BASE
            
            # Short: at BB upper + RSI overbought
            elif at_bb_upper and rsi[i] > 65:
                desired_signal = -SIZE_BASE
        
        # === EXIT LOGIC: Vol normalized (for vol spike entries) ===
        if in_position and entry_vol_spike and vol_normalized:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_VOLSPIKE * 0.9:
            final_signal = SIZE_VOLSPIKE
        elif desired_signal <= -SIZE_VOLSPIKE * 0.9:
            final_signal = -SIZE_VOLSPIKE
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.6
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                entry_vol_spike = is_vol_spike_entry
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                entry_vol_spike = is_vol_spike_entry
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                entry_vol_spike = False
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals