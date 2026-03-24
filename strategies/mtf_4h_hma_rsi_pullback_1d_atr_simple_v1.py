#!/usr/bin/env python3
"""
Experiment #251: 4h Primary + 1d/1w HTF — Simplified Trend Pullback

Hypothesis: After 200+ failed experiments, complex regime-switching (CHOP + CRSI + Donchian)
creates 0-trade scenarios. Return to proven simple trend-following with pullback entries:
- 1d HMA(21) for macro trend bias (proven in best strategies)
- 4h HMA(16/48) for medium-term trend direction
- RSI(14) pullback entries (35/65 thresholds - NOT extreme CRSI 15/85 which rarely triggers)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.25-0.30 (conservative for 4h volatility)

KEY INSIGHT FROM FAILURES:
- #240, #242, #248, #250: 0 trades due to too many confluence filters
- #243, #244, #247, #249: Negative Sharpe from regime-switching whipsaws
- CRSI <15 / >85 triggers TOO RARELY on 4h timeframe
- SOLUTION: Use RSI 35/65 (more frequent) + simpler trend filter

TARGET: 30-60 trades/year on 4h, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
CRITICAL: Entry conditions MUST trigger - RSI 35/65 hits ~20% of bars vs CRSI 15/85 at ~2%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d_atr_simple_v1"
timeframe = "4h"
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
    
    # Calculate 4h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 1d HMA for macro trend (aligned properly with shift(1))
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
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
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: bullish trend + RSI pullback to 35-50 zone
        rsi_pullback_long = (rsi_14[i] >= 35.0) and (rsi_14[i] <= 55.0)
        # Short: bearish trend + RSI pullback to 45-65 zone
        rsi_pullback_short = (rsi_14[i] >= 45.0) and (rsi_14[i] <= 65.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1d bullish + 4h bullish + RSI pullback
        if price_above_hma_1d and hma_bullish and rsi_pullback_long:
            desired_signal = POSITION_SIZE_FULL
        
        # SHORT ENTRY: 1d bearish + 4h bearish + RSI pullback
        elif price_below_hma_1d and hma_bearish and rsi_pullback_short:
            desired_signal = -POSITION_SIZE_FULL
        
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
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and hma_bearish:
            desired_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and hma_bullish:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        # Exit long if RSI becomes overbought (>70)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        # Exit short if RSI becomes oversold (<30)
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC - maintain position if setup still valid ===
        # Only hold if we're in position AND no exit signal triggered
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish (even if RSI moved)
                if hma_bullish and price_above_hma_1d:
                    desired_signal = POSITION_SIZE_HALF
            elif position_side < 0:
                # Hold short if trend still bearish (even if RSI moved)
                if hma_bearish and price_below_hma_1d:
                    desired_signal = -POSITION_SIZE_HALF
        
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