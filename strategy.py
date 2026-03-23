#!/usr/bin/env python3
"""
Experiment #262: 12h Primary + 1d/1w HTF — Simplified Multi-TF Trend Pullback

Hypothesis: 12h timeframe is underutilized. After #251 (4h HMA+RSI got Sharpe=-0.148),
the issue was hold logic creating fee churn. For 12h, use simpler logic:
- 1w HMA(21) for macro bias (very slow, reduces whipsaws)
- 1d HMA(16/48) crossover for intermediate trend confirmation
- 12h RSI(14) pullback entries at 30/70 (more frequent than 35/65)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.25-0.30 (conservative for 12h volatility)

KEY INSIGHT FROM #251 FAILURE:
- Hold logic with POSITION_SIZE_HALF created unnecessary signal changes
- SOLUTION: Binary signals only (0.0, ±0.30). Hold = maintain same signal value.
- Exit ONLY on: stoploss, trend reversal (HMA cross against), or RSI extreme (>75/<25)

TARGET: 20-50 trades/year on 12h, Sharpe > 0.5 on ALL symbols
CRITICAL: RSI 30/70 triggers ~15% of 12h bars vs 35/65 at ~25% - better balance
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d1w_atr_simple_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 1d HMA crossover (aligned properly with shift(1))
    hma_1d_16_raw = calculate_hma(df_1d['close'].values, 16)
    hma_1d_48_raw = calculate_hma(df_1d['close'].values, 48)
    hma_1d_16_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_16_raw)
    hma_1d_48_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_48_raw)
    
    # Calculate 1w HMA for macro bias (aligned properly with shift(1))
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_16_aligned[i]) or np.isnan(hma_1d_48_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND (HMA crossover) ===
        hma_1d_bullish = hma_1d_16_aligned[i] > hma_1d_48_aligned[i]
        hma_1d_bearish = hma_1d_16_aligned[i] < hma_1d_48_aligned[i]
        
        # === 12h TREND (HMA crossover) ===
        hma_12h_bullish = hma_16[i] > hma_48[i]
        hma_12h_bearish = hma_16[i] < hma_48[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: bullish trend + RSI pullback to 30-50 zone
        rsi_pullback_long = (rsi_14[i] >= 30.0) and (rsi_14[i] <= 50.0)
        # Short: bearish trend + RSI pullback to 50-70 zone
        rsi_pullback_short = (rsi_14[i] >= 50.0) and (rsi_14[i] <= 70.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1w bullish + 1d bullish + 12h bullish + RSI pullback
        if price_above_hma_1w and hma_1d_bullish and hma_12h_bullish and rsi_pullback_long:
            desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: 1w bearish + 1d bearish + 12h bearish + RSI pullback
        elif price_below_hma_1w and hma_1d_bearish and hma_12h_bearish and rsi_pullback_short:
            desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 12h trend turns bearish
        if in_position and position_side > 0 and hma_12h_bearish:
            desired_signal = 0.0
        
        # Exit short if 12h trend turns bullish
        if in_position and position_side < 0 and hma_12h_bullish:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        # Exit long if RSI becomes overbought (>75)
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            desired_signal = 0.0
        
        # Exit short if RSI becomes oversold (<25)
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        # CRITICAL: Only hold if trend conditions still met (no signal flip)
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if all trend conditions still bullish
                if price_above_hma_1w and hma_1d_bullish and hma_12h_bullish:
                    desired_signal = POSITION_SIZE
            elif position_side < 0:
                # Hold short if all trend conditions still bearish
                if price_below_hma_1w and hma_1d_bearish and hma_12h_bearish:
                    desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals