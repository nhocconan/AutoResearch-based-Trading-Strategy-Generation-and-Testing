#!/usr/bin/env python3
"""
Experiment #320: 1h Primary + 4h/12h HTF — Simplified Trend Pullback Strategy

Hypothesis: Recent 1h failures (#310, #315, #318) had 0 trades from over-filtering.
This version uses SIMPLE, PROVEN entry logic that GENERATES TRADES:
- 4h HMA(21) for trend direction (ONLY filter - no 12h, reduces complexity)
- 1h RSI(14) pullback entries (RSI<40 long in uptrend, RSI>60 short in downtrend)
- RSI cross confirmation for entry timing (avoids catching falling knife)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.25 (conservative for 1h timeframe)
- NO session/volume filters (these KILLED trades in #310, #318)

KEY CHANGES from failed 1h experiments:
- REMOVED session filter (8-20 UTC killed 70% of potential trades)
- REMOVED volume filter (killed trades during low-vol periods)
- REMOVED Choppiness Index (added complexity, reduced trades)
- SIMPLIFIED to pure trend + pullback (proven pattern from #299 success)
- LOOSEN RSI thresholds (40/60 vs 30/70) to trigger MORE entries
- Single HTF (4h only) instead of 4h+12h (less filtering = more trades)

TARGET: 40-80 trades/year on 1h, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_trend_pullback_4h_hma_rsi_atr_v1"
timeframe = "1h"
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
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Conservative for 1h timeframe
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # RSI cross tracking
    prev_rsi = rsi_14[0]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            prev_rsi = rsi_14[i]
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            prev_rsi = rsi_14[i]
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            prev_rsi = rsi_14[i]
            continue
        
        # === TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === RSI PULLBACK DETECTION ===
        # LONG: RSI < 40 (oversold pullback in uptrend) + crossing up
        rsi_cross_up = rsi_14[i] > prev_rsi and rsi_14[i-1] < rsi_14[i]
        rsi_cross_down = rsi_14[i] < prev_rsi and rsi_14[i-1] > rsi_14[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG entry: 4h bullish + 1h RSI pullback < 40 + RSI turning up
        if price_above_hma_4h and rsi_14[i] < 40.0 and rsi_cross_up:
            desired_signal = POSITION_SIZE
        
        # SHORT entry: 4h bearish + 1h RSI pullback > 60 + RSI turning down
        elif price_below_hma_4h and rsi_14[i] > 60.0 and rsi_cross_down:
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
        # Exit long if price crosses below 4h HMA
        if in_position and position_side > 0 and price_below_hma_4h:
            desired_signal = 0.0
        
        # Exit short if price crosses above 4h HMA
        if in_position and position_side < 0 and price_above_hma_4h:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        # Exit long when RSI > 75 (overbought)
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            desired_signal = 0.0
        
        # Exit short when RSI < 25 (oversold)
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC (maintain position unless exit trigger) ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Only hold if trend bias still supports position
            if position_side > 0 and price_above_hma_4h:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_hma_4h:
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
        prev_rsi = rsi_14[i]
    
    return signals