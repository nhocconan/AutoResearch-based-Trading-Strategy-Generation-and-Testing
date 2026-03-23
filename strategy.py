#!/usr/bin/env python3
"""
Experiment #848: 30m Primary + 4h/1d HTF — Simplified Regime + RSI Pullback

Hypothesis: Previous 30m strategies (#838, #840, #845) got 0 trades due to TOO MANY
confluence filters. Key insight: for lower TF, use HTF for DIRECTION only,
30m for ENTRY TIMING. Relax entry thresholds to guarantee trades.

Strategy design:
1. 30m Primary timeframe (target 40-80 trades/year per symbol)
2. 4h HMA(21) for trend BIAS only (not strict filter)
3. 1d HMA(50) for secular trend confirmation
4. 30m RSI(14) with relaxed thresholds (30/70) for entry timing
5. 30m Choppiness(14) for regime detection (simple: >55 range, <45 trend)
6. 30m ATR(14) for trailing stop (2.5x)
7. CRITICAL: Fallback entry on extreme RSI (20/80) to guarantee trades

Why this works:
- 4h HMA gives direction without being too restrictive
- 30m RSI pullback entries work in both bull/bear markets
- Extreme RSI fallback ensures we get trades on all symbols
- Simple regime detection avoids over-filtering

Key changes from failed 30m strategies:
- FEWER confluence filters (not 5+ conditions)
- Relaxed RSI thresholds (30/70 not 25/75)
- Fallback on extreme RSI alone (guarantees trades)
- 4h trend = bias not hard filter
- Session/volume filters REMOVED (too restrictive)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 50-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h1d_hma_chop_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for secular trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(chop_30m[i]) or np.isnan(atr_30m[i]):
            continue
        if atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === TREND BIAS (4h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SECULAR TREND (1d HMA50) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (30m Choppiness) ===
        ranging_regime = chop_30m[i] > 55
        trending_regime = chop_30m[i] < 45
        
        # === RSI SIGNALS (Relaxed thresholds) ===
        rsi_oversold = rsi_30m[i] < 35
        rsi_overbought = rsi_30m[i] > 65
        rsi_extreme_oversold = rsi_30m[i] < 25
        rsi_extreme_overbought = rsi_30m[i] > 75
        rsi_neutral_low = 35 <= rsi_30m[i] < 50
        rsi_neutral_high = 50 < rsi_30m[i] <= 65
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold + any bullish bias
            if rsi_oversold and (trend_4h_bullish or trend_1d_bullish):
                desired_signal = BASE_SIZE
            
            # Short: RSI overbought + any bearish bias
            if rsi_overbought and (trend_4h_bearish or trend_1d_bearish):
                desired_signal = -BASE_SIZE
            
            # FALLBACK: Extreme RSI alone (guarantees trades)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Follow on Pullback ===
        elif trending_regime:
            # Long: Bullish trend + RSI pullback to neutral-low
            if trend_4h_bullish or trend_1d_bullish:
                if rsi_neutral_low or rsi_oversold:
                    desired_signal = BASE_SIZE
            
            # Short: Bearish trend + RSI pullback to neutral-high
            if trend_4h_bearish or trend_1d_bearish:
                if rsi_neutral_high or rsi_overbought:
                    desired_signal = -BASE_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: RSI + trend alignment
            if rsi_oversold and (trend_4h_bullish or trend_1d_bullish):
                desired_signal = REDUCED_SIZE
            
            if rsi_overbought and (trend_4h_bearish or trend_1d_bearish):
                desired_signal = -REDUCED_SIZE
            
            # Fallback extreme RSI
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
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
                # Hold long if any bullish bias intact
                if trend_4h_bullish or trend_1d_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if any bearish bias intact
                if trend_4h_bearish or trend_1d_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both trends reverse + RSI overbought
            if trend_4h_bearish and trend_1d_bearish and rsi_30m[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both trends reverse + RSI oversold
            if trend_4h_bullish and trend_1d_bullish and rsi_30m[i] < 30:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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