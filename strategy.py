#!/usr/bin/env python3
"""
Experiment #196: 12h Primary + 1d HTF — HMA Trend + RSI Pullback + ATR Stop

Hypothesis: Higher timeframes (12h/1d) reduce noise and whipsaws seen in lower TF strategies.
Many 4h strategies failed due to excessive trading in 2022 crash. 12h should capture
major trends while avoiding chop. Using simple, proven components:

1. 1d HMA(21) = macro trend bias (long only above, short only below)
2. 12h RSI(14) = pullback entry trigger (35-45 for long, 55-65 for short)
3. ATR(14) * 2.5 = trailing stoploss
4. Discrete position sizing: 0.0, ±0.30

Key insight from failures: Too many filters = 0 trades. Keep entry conditions
loose enough to generate 25-40 trades/year while maintaining trend filter.

TARGET: 25-40 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
Position sizing: 0.30 max (30% of capital), discrete levels to minimize fee churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_pullback_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average - smoother than EMA, less lag than SMA."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 1d HMA for macro trend (aligned with shift(1) for no look-ahead)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30  # 30% of capital - discrete level
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === MACRO TREND BIAS (1d HMA) ===
        bullish_trend = close[i] > hma_1d_aligned[i]
        bearish_trend = close[i] < hma_1d_aligned[i]
        
        # === ENTRY LOGIC (12h RSI pullback in direction of trend) ===
        new_signal = 0.0
        
        # Long: bullish 1d trend + RSI pullback to 35-45 zone
        if bullish_trend and 35.0 <= rsi_14[i] <= 45.0:
            new_signal = POSITION_SIZE
        
        # Short: bearish 1d trend + RSI bounce to 55-65 zone
        elif bearish_trend and 55.0 <= rsi_14[i] <= 65.0:
            new_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 1d trend flips bearish
        if in_position and position_side > 0 and bearish_trend:
            new_signal = 0.0
        
        # Exit short if 1d trend flips bullish
        if in_position and position_side < 0 and bullish_trend:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip (long to short or vice versa)
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals