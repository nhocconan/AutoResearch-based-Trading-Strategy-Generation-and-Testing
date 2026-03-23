#!/usr/bin/env python3
"""
Experiment #258: 30m Primary + 4h/1d HTF — Simplified Trend Pullback

Hypothesis: After 200+ failed experiments, the key lesson is:
1. Too many confluence filters = 0 trades (#248, #250, #255 all failed with Sharpe=0)
2. Complex regime-switching = whipsaws (#249, #252, #253, #254 all negative Sharpe)
3. SIMPLE trend + pullback works best (proven in #251 baseline)

For 30m timeframe, I will use:
- 1d HMA(21) for MACRO trend bias (only trade in direction of daily trend)
- 4h HMA(16/48) for MEDIUM-TERM trend confirmation
- 30m RSI(14) pullback entries (40/60 thresholds - NOT extreme values)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.25 (conservative for 30m volatility)

KEY INSIGHT: Lower TF needs FEWER filters, not more. Session/volume/z-score filters
caused 0 trades in previous experiments. Use HTF trend as the main filter.

TARGET: 40-80 trades/year on 30m, Sharpe > 0.5 on ALL symbols
CRITICAL: Entry conditions MUST trigger - RSI 40/60 hits ~30% of bars
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h1d_atr_simple_v1"
timeframe = "30m"
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 4h HMA for medium-term trend (aligned properly with shift(1))
    hma_4h_16_raw = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48_raw = calculate_hma(df_4h['close'].values, 48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16_raw)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48_raw)
    
    # Calculate 1d HMA for macro trend (aligned properly with shift(1))
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25
    
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
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === RSI PULLBACK SIGNALS (30m) ===
        # Long: bullish trend + RSI pullback to 40-55 zone
        rsi_pullback_long = (rsi_14[i] >= 40.0) and (rsi_14[i] <= 55.0)
        # Short: bearish trend + RSI pullback to 45-60 zone
        rsi_pullback_short = (rsi_14[i] >= 45.0) and (rsi_14[i] <= 60.0)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1d bullish + 4h bullish + RSI pullback
        if price_above_hma_1d and hma_4h_bullish and rsi_pullback_long:
            desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: 1d bearish + 4h bearish + RSI pullback
        elif price_below_hma_1d and hma_4h_bearish and rsi_pullback_short:
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
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and hma_4h_bearish:
            desired_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and hma_4h_bullish:
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
                # Hold long if trend still bullish
                if hma_4h_bullish and price_above_hma_1d:
                    desired_signal = POSITION_SIZE
            elif position_side < 0:
                # Hold short if trend still bearish
                if hma_4h_bearish and price_below_hma_1d:
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