#!/usr/bin/env python3
"""
Experiment #1095: 1h Primary + 4h/1d HTF — Simple Trend Pullback with Volume

Hypothesis: After 794+ failed experiments, the key insight is:
1. Complex regime-switching (Choppiness + CRSI) leads to 0 trades on lower TF
2. SIMPLER is better: 1d HMA for macro trend + 4h RSI for pullback entries
3. 1h timeframe needs LOOSE entry conditions to generate 30-60 trades/year
4. Volume filter must be lenient (0.8x avg, not 1.2x) to avoid filtering all trades
5. Session filter REMOVED — reduces trades too much on 1h TF

Why this should beat Sharpe=0.000 (previous 1h strategies):
- Simpler entry logic = more trades (not 0 trades like #1085, #1088, #1090)
- Lenient RSI thresholds (35/65 not 30/70) ensure adequate frequency
- Volume at 0.8x avg (not 1.2x) confirms without over-filtering
- 1d HMA macro filter provides edge without being too restrictive

Timeframe: 1h (primary)
HTF: 4h and 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    
    Formula:
    1. WMA1 = WMA(close, period/2)
    2. WMA2 = WMA(close, period)
    3. WMA3 = WMA(2*WMA1 - WMA2, sqrt(period))
    4. HMA = WMA3
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = int(period / 2)
    if half < 1:
        half = 1
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    # 2*WMA1 - WMA2
    diff = 2 * wma1 - wma2
    
    # WMA of diff with sqrt(period)
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    hma = wma(diff, sqrt_period)
    return hma

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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume for volume confirmation."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 4h RSI for pullback entries
    rsi_4h_raw = calculate_rsi(df_4h['close'].values, period=14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_raw)
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi_4h_aligned[i]):
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10 or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PULLBACK SIGNAL (4h RSI) ===
        # Lenient thresholds to ensure trades
        rsi_4h_oversold = rsi_4h_aligned[i] < 45.0
        rsi_4h_overbought = rsi_4h_aligned[i] > 55.0
        
        # === VOLUME CONFIRMATION (lenient) ===
        vol_ok = volume[i] > 0.8 * vol_sma[i]
        
        # === 1h RSI for entry timing ===
        rsi_1h_oversold = rsi_1h[i] < 40.0
        rsi_1h_overbought = rsi_1h[i] > 60.0
        
        # === VOLATILITY CHECK ===
        vol_spike = atr[i] > 1.5 * np.nanmedian(atr[max(0, i-100):i]) if i > 100 else False
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + 4h RSI pullback + volume ok + 1h RSI confirms
        if macro_bull and rsi_4h_oversold and vol_ok:
            if rsi_1h_oversold or rsi_1h[i] < 50.0:
                desired_signal = current_size
        
        # === SHORT ENTRY ===
        # Macro bear + 4h RSI pullback + volume ok + 1h RSI confirms
        elif macro_bear and rsi_4h_overbought and vol_ok:
            if rsi_1h_overbought or rsi_1h[i] > 50.0:
                desired_signal = -current_size
        
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
                # Hold long if macro still bull
                if macro_bull:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if macro still bear
                if macro_bear:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses
            if macro_bear and rsi_1h[i] > 65.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses
            if macro_bull and rsi_1h[i] < 35.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
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