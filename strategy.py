#!/usr/bin/env python3
"""
Experiment #458: 30m Primary + 4h/1d HTF — Simplified Trend Pullback with Volume

Hypothesis: Previous 30m strategies (#448, #455) failed with Sharpe=0.000 due to TOO STRICT
entry conditions (session filters + multiple confluence = 0 trades). This version:
1. Uses 4h HMA(21) for PRIMARY trend direction (long-only when 4h HMA up, short when down)
2. Uses 30m RSI(14) pullback for entry timing (RSI<40 in uptrend, RSI>60 in downtrend)
3. Uses volume filter (volume > 0.7x 20-bar avg) — simpler than session filter
4. Uses 1d HMA(50) for ultra-long-term bias filter (only trade with 1d trend)
5. ATR(14) trailing stop at 2.5x for risk management
6. Position size: 0.25 base, 0.30 on strong confluence, discrete levels

Key change from #448/#455: REMOVED session filter, simplified RSI thresholds (40/60 not 25/75),
removed Choppiness Index (was too restrictive). This should generate 40-80 trades/year.

Target: Sharpe > 0.612, 40-80 trades over 4-year train, DD < -35%
Timeframe: 30m (lower TF for entry timing within HTF trend)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h1d_volume_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    volume_sma20 = calculate_sma(volume, 20)
    
    # Calculate and align HTF HMA for bias (4h and 1d)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% base position size for 30m
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(volume_sma20[i]) or volume_sma20[i] <= 0:
            continue
        
        # === HTF TREND BIAS (4h HMA21) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === ULTRA-LONG TERM BIAS (1d HMA50) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLUME FILTER (simpler than session) ===
        volume_ratio = volume[i] / (volume_sma20[i] + 1e-10)
        volume_ok = volume_ratio > 0.7  # At least 70% of avg volume
        
        # === RSI PULLBACK SIGNALS ===
        # In uptrend: wait for RSI pullback to 40-50 zone
        rsi_pullback_long = rsi_14[i] < 45.0 and rsi_14[i] > 25.0
        # In downtrend: wait for RSI rally to 55-65 zone
        rsi_pullback_short = rsi_14[i] > 55.0 and rsi_14[i] < 75.0
        
        # === EXTREME RSI (stronger signal) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5  # Reduce size in extreme vol
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        # === LONG SETUP: 4h bullish + 1d bullish + RSI pullback ===
        if price_above_hma_4h and price_above_hma_1d and volume_ok:
            signal_strength = 1
            
            # RSI pullback entry
            if rsi_pullback_long:
                signal_strength += 2
            
            # Extreme oversold = stronger signal
            if rsi_oversold:
                signal_strength += 1
            
            # Need at least strength 2 to enter
            if signal_strength >= 2:
                desired_signal = position_size * (0.8 + 0.2 * min(signal_strength, 4) / 4)
        
        # === SHORT SETUP: 4h bearish + 1d bearish + RSI rally ===
        elif price_below_hma_4h and price_below_hma_1d and volume_ok:
            signal_strength = 1
            
            # RSI rally entry
            if rsi_pullback_short:
                signal_strength += 2
            
            # Extreme overbought = stronger signal
            if rsi_overbought:
                signal_strength += 1
            
            # Need at least strength 2 to enter
            if signal_strength >= 2:
                desired_signal = -position_size * (0.8 + 0.2 * min(signal_strength, 4) / 4)
        
        # === CAP SIGNAL TO MAX 0.35 ===
        if desired_signal > 0.35:
            desired_signal = 0.35
        elif desired_signal < -0.35:
            desired_signal = -0.35
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === RSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if HTF trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_4h:
                desired_signal = position_size
            elif position_side < 0 and price_below_hma_4h:
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.22:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.22:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.15
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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