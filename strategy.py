#!/usr/bin/env python3
"""
Experiment #779: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + Choppiness Regime

Hypothesis: After 500+ failed strategies, I'm simplifying drastically:
1. CRSI is overused (100+ failures) — use simpler RSI(14) with clearer thresholds
2. Choppiness Index works best as simple regime filter (CHOP>60=range, <40=trend)
3. 1d HMA(21) provides cleaner trend bias than EMA (less lag, proven in best strategies)
4. Entry conditions too strict in recent failures — LOOSEN for trade frequency
5. ATR-based sizing and stops control drawdown better than fixed stops

Strategy design:
1. 1d HMA(21) for trend bias (aligned via mtf_data helper)
2. 4h RSI(14) for entry timing (30/70 thresholds, not extreme)
3. 4h Choppiness(14) for regime (simple binary: >60=range, <40=trend)
4. 4h ATR(14) for trailing stop (2.5x) and position sizing
5. Discrete signals: 0.0, ±0.25, ±0.30
6. Position sizing: 0.25-0.30 (conservative for drawdown control)

Key differences from failed attempts:
- Simpler RSI(14) vs complex CRSI (more reliable, less overfitting)
- Looser RSI thresholds (30/70 vs 10/90) for MORE trades
- Single HTF (1d) vs multiple (1d+1w) for cleaner signals
- HMA(21) vs EMA(50) for faster trend response on 4h entries

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_chop_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    s = pd.Series(series)
    
    # WMA helper
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return data.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    half = period // 2
    wma_half = wma(s, half)
    wma_full = wma(s, period)
    
    diff = 2 * wma_half - wma_full
    hma = wma(diff, int(np.sqrt(period)))
    
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
    Choppiness Index — measures market consolidation vs trending.
    CHOP > 61.8 = ranging/consolidating
    CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop_raw, 0, 100)
    return chop

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
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Also calculate 4h HMA for additional confirmation
    hma_4h = calculate_hma(close, 21)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h[i]):
            continue
        if np.isnan(chop_4h[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA CONFIRMATION ===
        trend_4h_bullish = close[i] > hma_4h[i]
        trend_4h_bearish = close[i] < hma_4h[i]
        
        # === REGIME DETECTION (Choppiness) ===
        # CHOP > 60 = ranging, CHOP < 40 = trending
        ranging_regime = chop_4h[i] > 60
        trending_regime = chop_4h[i] < 40
        neutral_regime = not ranging_regime and not trending_regime
        
        # === RSI SIGNALS (LOOSE thresholds for trade frequency) ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_extreme_oversold = rsi_4h[i] < 25
        rsi_extreme_overbought = rsi_4h[i] > 75
        rsi_neutral_low = 35 < rsi_4h[i] < 50
        rsi_neutral_high = 50 < rsi_4h[i] < 65
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (CHOP < 40) ===
        if trending_regime:
            # Trend following long: 1d bullish + 4h bullish + RSI pullback
            if trend_1d_bullish and trend_4h_bullish and rsi_neutral_low:
                desired_signal = BASE_SIZE
            
            # Trend following short: 1d bearish + 4h bearish + RSI pullback
            if trend_1d_bearish and trend_4h_bearish and rsi_neutral_high:
                desired_signal = -BASE_SIZE
            
            # Strong trend continuation: extreme RSI in trend direction
            if trend_1d_bullish and trend_4h_bullish and rsi_4h[i] > 55:
                desired_signal = REDUCED_SIZE
            
            if trend_1d_bearish and trend_4h_bearish and rsi_4h[i] < 45:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME LOGIC (CHOP > 60) ===
        elif ranging_regime:
            # Mean reversion long: RSI oversold + price near support
            if rsi_oversold and not trend_1d_bearish:
                desired_signal = REDUCED_SIZE
            
            # Mean reversion short: RSI overbought + price near resistance
            if rsi_overbought and not trend_1d_bullish:
                desired_signal = -REDUCED_SIZE
            
            # Extreme mean reversion: ignore trend, just RSI extremes
            if rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (40 <= CHOP <= 60) ===
        else:
            # Conservative: only trade with 1d trend + RSI confirmation
            if trend_1d_bullish and rsi_oversold:
                desired_signal = REDUCED_SIZE
            
            if trend_1d_bearish and rsi_overbought:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d trend intact and RSI not overbought
                if trend_1d_bullish and rsi_4h[i] < 75:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 1d trend intact and RSI not oversold
                if trend_1d_bearish and rsi_4h[i] > 25:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses
            if trend_1d_bearish and rsi_4h[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses
            if trend_1d_bullish and rsi_4h[i] < 40:
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