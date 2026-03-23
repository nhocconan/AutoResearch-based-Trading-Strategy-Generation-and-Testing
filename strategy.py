#!/usr/bin/env python3
"""
Experiment #704: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + RSI Momentum + Volume Confirmation

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency better than EMA/HMA,
reducing whipsaw in choppy markets while capturing trends efficiently. Combined with RSI(7) momentum
(not extremes), ADX trend strength filter, and volume confirmation, this should generate 30-50 trades/year
with positive Sharpe across ALL symbols (BTC/ETH/SOL).

Key Differences from Failed Strategies:
1. KAMA instead of HMA/EMA - adapts to volatility regime automatically
2. RSI(7) with 45/55 thresholds (not 30/70 extremes) - more trades, less waiting
3. Volume ratio confirmation - filters false breakouts
4. ADX(14) > 18 filter - only trade when trend has minimum strength
5. Asymmetric entry: long bias when 12h HMA bullish, short bias when 12h HMA bearish
6. 2.5x ATR trailing stop - protects against reversals

Why this should work:
- KAMA proven in "Trading Systems and Methods" (Kaufman) for adaptive trend following
- 4h TF worked in current best (Sharpe=0.612) - stay with proven timeframe
- RSI(7) more responsive than RSI(14) for 4h entries
- Volume confirmation reduces false signals that killed #692/#694
- Less strict than CRSI approaches that generated 0 trades (#699, #700, #703)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_volume_adx_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts to market efficiency.
    Reference: Perry Kaufman, "Trading Systems and Methods"
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[max(0, i - period):i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = sc ** 2  # Square for smoother adaptation
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = close[i]
    
    return kama

def calculate_rsi(close, period=7):
    """Relative Strength Index - faster period for 4h entries."""
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100 * plus_dm_smooth / (atr + 1e-10)
        di_minus = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, atr

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss."""
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

def calculate_hma(series, period):
    """Hull Moving Average for HTF trend bias."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi_4h = calculate_rsi(close, period=7)
    adx_4h, atr_4h = calculate_adx(high, low, close, period=14)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_sma + 1e-10)
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Need buffer for indicators
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]):
            continue
        if np.isnan(adx_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (HTF HMA) ===
        trend_bullish_12h = close[i] > hma_12h_aligned[i]
        trend_bearish_12h = close[i] < hma_12h_aligned[i]
        trend_bullish_1d = close[i] > hma_1d_aligned[i]
        trend_bearish_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both HTF agree
        trend_strong_bullish = trend_bullish_12h and trend_bullish_1d
        trend_strong_bearish = trend_bearish_12h and trend_bearish_1d
        
        # === KAMA TREND (Primary) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === RSI MOMENTUM ===
        rsi_momentum_long = rsi_4h[i] > 50  # Bullish momentum
        rsi_momentum_short = rsi_4h[i] < 50  # Bearish momentum
        
        # === ADX STRENGTH ===
        adx_strong = adx_4h[i] > 18  # Minimum trend strength
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2  # 20% above average
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # Reduce size in high volatility
        atr_ratio = atr_4h[i] / (np.nanmedian(atr_4h[max(0, i-50):i+1]) + 1e-10)
        if atr_ratio > 2.0:
            current_size = REDUCED_SIZE
        
        # === LONG ENTRY ===
        # Primary: KAMA bullish + RSI momentum + ADX strength
        long_condition = (kama_bullish and rsi_momentum_long and adx_strong)
        
        # Add HTF bias bonus
        if long_condition:
            if trend_strong_bullish and volume_confirmed:
                desired_signal = current_size  # Full size with all confirmations
            elif trend_bullish_12h and volume_confirmed:
                desired_signal = current_size * 0.8  # Reduced size
            elif trend_bullish_12h:
                desired_signal = current_size * 0.6  # Minimal HTF confirmation
            elif volume_confirmed:
                desired_signal = current_size * 0.5  # Volume only
        
        # === SHORT ENTRY ===
        # Primary: KAMA bearish + RSI momentum + ADX strength
        short_condition = (kama_bearish and rsi_momentum_short and adx_strong)
        
        if short_condition:
            if trend_strong_bearish and volume_confirmed:
                desired_signal = -current_size  # Full size with all confirmations
            elif trend_bearish_12h and volume_confirmed:
                desired_signal = -current_size * 0.8  # Reduced size
            elif trend_bearish_12h:
                desired_signal = -current_size * 0.6  # Minimal HTF confirmation
            elif volume_confirmed:
                desired_signal = -current_size * 0.5  # Volume only
        
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
                # Hold long if KAMA still bullish and RSI not overbought
                if kama_bullish and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA still bearish and RSI not oversold
                if kama_bearish and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: KAMA reverses OR RSI overbought
        if in_position and position_side > 0:
            if kama_bearish and rsi_4h[i] > 65:
                desired_signal = 0.0
            elif close[i] < hma_12h_aligned[i] and close[i] < hma_1d_aligned[i]:
                desired_signal = 0.0  # HTF trend reversal
        
        # Short exit: KAMA reverses OR RSI oversold
        if in_position and position_side < 0:
            if kama_bullish and rsi_4h[i] < 35:
                desired_signal = 0.0
            elif close[i] > hma_12h_aligned[i] and close[i] > hma_1d_aligned[i]:
                desired_signal = 0.0  # HTF trend reversal
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.9:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.9:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.9:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.9:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        
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