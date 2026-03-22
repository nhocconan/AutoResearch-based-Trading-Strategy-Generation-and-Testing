#!/usr/bin/env python3
"""
Experiment #359: 4h Primary + 1d HTF — Simplified Trend-Follow with RSI Pullback

Hypothesis: After 358 experiments, the pattern is clear:
1. Complex dual-regime strategies with Choppiness Index are FAILING (Sharpe negative or 0 trades)
2. Over-filtered entry conditions = 0 trades = auto-reject (see #348, #350, #355)
3. SIMPLE trend-follow works: 4h HMA crossover + 1d HMA bias + RSI pullback entry
4. LOOSE entry filters generate trades; strict filters generate nothing
5. ATR trailing stop cuts losers; position sizing 0.25-0.30 controls drawdown

Why this might beat Sharpe=0.435:
- Fewer confluence filters = MORE trades (avoiding 0-trade failure mode)
- 4h HMA(16/48) crossover is proven trend indicator
- 1d HMA(21) alignment prevents counter-trend trades
- RSI 40-60 pullback zone (not extremes) = frequent entry opportunities
- ADX > 18 (not 25+) = more trending periods captured
- 2.5x ATR trailing stop protects capital

Position sizing: 0.30 long, 0.25 short (asymmetric for crypto long bias)
Target: 30-60 trades/year on 4h timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_1d_trend_simp_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100.0 * minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        new_signal = 0.0
        
        # === TREND DIRECTION (4h HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === 1D MAJOR TREND BIAS ===
        trend_1d_bull = close[i] > hma_1d_21_aligned[i]
        trend_1d_bear = close[i] < hma_1d_21_aligned[i]
        
        # === ADX TREND STRENGTH (loose filter: >18 not >25) ===
        trend_strong = adx_14[i] > 18.0
        
        # === RSI PULLBACK ENTRY (loose zone: 40-60, not extremes) ===
        # Long: RSI pulled back to 40-58 in uptrend
        rsi_long_entry = 40.0 <= rsi_14[i] <= 58.0
        # Short: RSI rallied to 42-60 in downtrend
        rsi_short_entry = 42.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC (LOOSE - generate trades!) ===
        # Long: 4h bullish + 1d bullish/neutral + ADX strong + RSI pullback
        if hma_bullish and (trend_1d_bull or not trend_1d_bear) and trend_strong and rsi_long_entry:
            new_signal = LONG_SIZE
        
        # Short: 4h bearish + 1d bearish/neutral + ADX strong + RSI pullback
        if hma_bearish and (trend_1d_bear or not trend_1d_bull) and trend_strong and rsi_short_entry:
            if new_signal == 0.0:  # Don't override long signal
                new_signal = -SHORT_SIZE
        
        # === STOPLOSS (Rule 6 - 2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long
                if highest_price == 0.0 or close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side != 0:
            if position_side > 0 and hma_bearish:
                new_signal = 0.0
            if position_side < 0 and hma_bullish:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals