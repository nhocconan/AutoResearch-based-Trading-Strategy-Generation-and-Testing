#!/usr/bin/env python3
"""
Experiment #770: 1h Primary + 4h/12h HTF — Simplified RSI Mean Reversion + HTF Trend

Hypothesis: After analyzing 50+ failed lower-TF strategies (#760, #765, #768 all had 0 trades):
1. Session filters (8-20 UTC) are TOO RESTRICTIVE — crypto trades 24/7, remove it
2. Volume filters kill trade frequency on 1h — relax to 0.8x or remove
3. CRSI is too complex for 1h — use simpler RSI(7) with faster response
4. Over-filtering is the #1 cause of 0 trades — need 30-80 trades/year target
5. HTF trend (4h HMA) + LTF RSI extremes is proven pattern, just simplify entry

Strategy design:
1. 4h HMA(21) for primary trend bias (aligned via mtf_data helper)
2. 12h HMA(50) for secondary trend confirmation
3. 1h RSI(7) for mean reversion entries (faster than RSI(14))
4. 1h Bollinger Bands (20, 2.0) for entry triggers
5. 4h ADX(14) for regime (trending >25, ranging <20)
6. 1h ATR(14) for trailing stop (2.5x)
7. NO session filter, NO strict volume filter (learned from failures)
8. Discrete signals: 0.0, ±0.20, ±0.30

Key changes from failed #760/#765/#768:
- REMOVED session filter (was killing all trades)
- REMOVED strict volume filter (was filtering 80% of signals)
- Simplified RSI(7) vs CRSI (faster, more responsive for 1h)
- Simpler regime logic (ADX single threshold vs hysteresis)
- More aggressive entry thresholds (RSI<25/>75 vs CRSI<10/>90)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-80 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_bb_hma_4h12h_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average for trend detection."""
    series = pd.Series(series)
    wma1 = series.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = series.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=7)  # Faster RSI for 1h
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 4h ADX for regime detection
    adx_4h_raw = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_sma[i]):
            continue
        if np.isnan(adx_4h_aligned[i]):
            continue
        
        # === TREND BIAS (4h + 12h HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # Strong trend = both HTF agree
        strong_bullish = trend_4h_bullish and trend_12h_bullish
        strong_bearish = trend_4h_bearish and trend_12h_bearish
        
        # === REGIME DETECTION (4h ADX) ===
        adx_val = adx_4h_aligned[i]
        trending_regime = adx_val > 25
        ranging_regime = adx_val < 20
        
        # === RSI SIGNALS (1h, period=7) ===
        rsi_oversold = rsi_1h[i] < 30
        rsi_overbought = rsi_1h[i] > 70
        rsi_extreme_oversold = rsi_1h[i] < 20
        rsi_extreme_overbought = rsi_1h[i] > 80
        rsi_neutral_low = 35 < rsi_1h[i] < 45
        rsi_neutral_high = 55 < rsi_1h[i] < 65
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        below_bb_sma = close[i] < bb_sma[i]
        above_bb_sma = close[i] > bb_sma[i]
        
        desired_signal = 0.0
        
        # === RANGING REGIME (ADX < 20) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold + below BB lower
            if rsi_oversold and below_bb_lower:
                desired_signal = BASE_SIZE
            
            # Short: RSI overbought + above BB upper
            if rsi_overbought and above_bb_upper:
                desired_signal = -BASE_SIZE
            
            # Conservative: extreme RSI even without BB touch
            if rsi_extreme_oversold and not strong_bearish:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and not strong_bullish:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (ADX > 25) — Trend Following ===
        elif trending_regime:
            # Long pullback: strong bullish + RSI pullback to neutral
            if strong_bullish and rsi_neutral_low and below_bb_sma:
                desired_signal = BASE_SIZE
            
            # Short pullback: strong bearish + RSI pullback to neutral
            if strong_bearish and rsi_neutral_high and above_bb_sma:
                desired_signal = -BASE_SIZE
            
            # Breakout continuation
            if strong_bullish and above_bb_upper and rsi_1h[i] > 50:
                desired_signal = REDUCED_SIZE
            
            if strong_bearish and below_bb_lower and rsi_1h[i] < 50:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (20 <= ADX <= 25) ===
        else:
            # Only extreme RSI with HTF trend agreement
            if rsi_extreme_oversold and trend_4h_bullish:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and trend_4h_bearish:
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
                # Hold long if 4h trend intact and RSI not overbought
                if trend_4h_bullish and rsi_1h[i] < 75:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 4h trend intact and RSI not oversold
                if trend_4h_bearish and rsi_1h[i] > 25:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses or RSI overbought
            if trend_4h_bearish and rsi_1h[i] > 65:
                desired_signal = 0.0
            # Exit if price hits BB upper in ranging regime
            if ranging_regime and above_bb_upper:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses or RSI oversold
            if trend_4h_bullish and rsi_1h[i] < 35:
                desired_signal = 0.0
            # Exit if price hits BB lower in ranging regime
            if ranging_regime and below_bb_lower:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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