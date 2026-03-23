#!/usr/bin/env python3
"""
Experiment #714: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + RSI Pullback

Hypothesis: KAMA (Kaufman Adaptive Moving Average) outperforms HMA/EMA in crypto
because it adapts to market efficiency ratio - faster in trends, slower in chop.
Combined with 12h HMA trend bias and loosened RSI entries to ensure trade frequency.

Key differences from #709 (Sharpe=-0.411):
1. KAMA instead of HMA for primary trend (adapts to volatility regimes)
2. Add 12h HTF (not just 1d) for intermediate trend confirmation
3. Looser RSI thresholds (40/60 not 45/55) to generate more trades
4. Simpler regime logic - avoid over-filtering that caused 0 trades in #705/#710
5. Entry on RSI cross through threshold (not just below/above) for better timing

Why this should beat Sharpe=0.612:
- KAMA proven in literature to reduce whipsaw in ranging markets
- 12h HTF gives smoother trend signal than 1d alone
- Looser entries ensure 30-50 trades/year target on 4h
- ATR trailing stop protects from 2022-style crashes

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_pullback_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts to market efficiency.
    ER (Efficiency Ratio) determines smoothing constant.
    High ER (trending) = faster response, Low ER (chop) = slower response.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[period] = close[period]
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    
    if n < period * 2:
        return adx, atr
    
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
    atr_vals = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        di_plus = 100 * plus_dm_smooth / (atr_vals + 1e-10)
        di_minus = 100 * minus_dm_smooth / (atr_vals + 1e-10)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr[:] = atr_vals
    
    return adx, atr

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr_vals = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr[:] = atr_vals
    
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi_4h = calculate_rsi(close, period=14)
    adx_4h, atr_4h = calculate_adx(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMAs for trend bias
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
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or np.isnan(kama_4h[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(adx_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (HTF HMAs) ===
        # Bullish: price above both 12h and 1d HMA
        trend_bullish_strong = close[i] > hma_12h_aligned[i] and close[i] > hma_1d_aligned[i]
        trend_bullish_weak = close[i] > hma_12h_aligned[i]
        
        # Bearish: price below both 12h and 1d HMA
        trend_bearish_strong = close[i] < hma_12h_aligned[i] and close[i] < hma_1d_aligned[i]
        trend_bearish_weak = close[i] < hma_12h_aligned[i]
        
        # === KAMA TREND (Primary 4h) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ADX REGIME ===
        adx_strong = adx_4h[i] > 25  # Trending
        adx_weak = adx_4h[i] < 20    # Ranging
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRIES ===
        # Strong long: All trend alignments + RSI pullback
        if trend_bullish_strong and kama_bullish and above_sma200:
            if rsi_4h[i] < 45:  # Pullback entry
                desired_signal = current_size
            elif rsi_4h[i] < 55 and rsi_4h[i-1] >= 55:  # RSI cross below 55
                desired_signal = REDUCED_SIZE
        
        # Moderate long: 12h HMA + KAMA + RSI oversold
        elif trend_bullish_weak and kama_bullish:
            if rsi_4h[i] < 40:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        # Strong short: All trend alignments + RSI bounce
        if trend_bearish_strong and kama_bearish and below_sma200:
            if rsi_4h[i] > 55:  # Bounce entry
                desired_signal = -current_size
            elif rsi_4h[i] > 45 and rsi_4h[i-1] <= 45:  # RSI cross above 45
                desired_signal = -REDUCED_SIZE
        
        # Moderate short: 12h HMA + KAMA + RSI overbought
        elif trend_bearish_weak and kama_bearish:
            if rsi_4h[i] > 60:
                desired_signal = -REDUCED_SIZE
        
        # === RANGE REGIME (ADX < 20) - Mean Reversion ===
        if adx_weak:
            # Long: RSI very oversold in range
            if rsi_4h[i] < 30 and kama_bullish:
                desired_signal = REDUCED_SIZE
            # Short: RSI very overbought in range
            elif rsi_4h[i] > 70 and kama_bearish:
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if RSI not overbought and KAMA intact
                if rsi_4h[i] < 70 and close[i] > kama_4h[i]:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if RSI not oversold and KAMA intact
                if rsi_4h[i] > 30 and close[i] < kama_4h[i]:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long on RSI overbought or KAMA cross
            if rsi_4h[i] > 75:
                desired_signal = 0.0
            elif close[i] < kama_4h[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short on RSI oversold or KAMA cross
            if rsi_4h[i] < 25:
                desired_signal = 0.0
            elif close[i] > kama_4h[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.8 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.8 else -REDUCED_SIZE
        else:
            desired_signal = 0.0
        
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