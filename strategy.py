#!/usr/bin/env python3
"""
EXPERIMENT #002 - KAMA Adaptive Trend + RSI Pullback with 4h Filter (30m)
=========================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market regimes
better than fixed EMAs. On 30m timeframe, we enter on RSI pullbacks in the
direction of the 4h KAMA trend. This captures medium-term trends while
avoiding chasing breakouts at extremes.

Key features:
- Primary TF: 30m (balances signal frequency vs noise)
- HTF filter: 4h KAMA(21) for trend direction
- Entry: RSI(14) pullback to 40-50 in uptrend, 50-60 in downtrend
- Confirmation: KAMA slope + price above/below KAMA
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels

Why different from failed strategies:
- KAMA adapts to volatility (unlike fixed Supertrend)
- RSI pullback entries (not breakout = more trades than Donchian)
- 30m TF = more signals than 1d, less noise than 15m
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_pullback_4hfilter_30m_v1"
timeframe = "30m"
leverage = 1.0


def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Change = absolute price change over efficiency period
    change = np.abs(close - np.roll(close, efficiency_period))
    change[:efficiency_period] = np.nan
    
    # Volatility = sum of absolute single-period changes
    volatility = np.zeros(n)
    for i in range(efficiency_period, n):
        volatility[i] = np.sum(np.abs(close[i-efficiency_period+1:i+1] - 
                                       np.roll(close[i-efficiency_period+1:i+1], 1))[1:])
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    er[volatility > 0] = change[volatility > 0] / volatility[volatility > 0]
    er[:efficiency_period] = np.nan
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Dynamic smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    valid_start = efficiency_period
    kama[valid_start] = close[valid_start]
    
    # Calculate KAMA recursively
    for i in range(valid_start + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_kama_slope(kama, lookback=5):
    """Calculate KAMA slope (rate of change)"""
    slope = np.zeros(len(kama))
    slope[:] = np.nan
    for i in range(lookback, len(kama)):
        if kama[i] > 0 and kama[i-lookback] > 0:
            slope[i] = (kama[i] - kama[i-lookback]) / kama[i-lookback]
    return slope


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    kama_4h = calculate_kama(df_4h['close'].values, efficiency_period=10, fast_period=2, slow_period=30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)  # auto shift(1)
    
    # Calculate 30m indicators
    kama_30m = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    kama_slope_30m = calculate_kama_slope(kama_30m, lookback=5)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    
    min_period = 50  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(kama_30m[i]) or 
            np.isnan(kama_slope_30m[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h Trend filter (HTF)
        htf_trend = 0
        if close[i] > kama_4h_aligned[i]:
            htf_trend = 1  # Bullish HTF
        elif close[i] < kama_4h_aligned[i]:
            htf_trend = -1  # Bearish HTF
        
        # 30m KAMA trend confirmation
        ltf_trend = 0
        if kama_slope_30m[i] > 0.001 and close[i] > kama_30m[i]:
            ltf_trend = 1  # Bullish LTF
        elif kama_slope_30m[i] < -0.001 and close[i] < kama_30m[i]:
            ltf_trend = -1  # Bearish LTF
        
        # RSI pullback entry logic
        # In uptrend: buy when RSI pulls back to 40-50 (not oversold, just dip)
        # In downtrend: sell when RSI rallies to 50-60 (not overbought, just bounce)
        rsi_long_signal = 40 <= rsi[i] <= 55
        rsi_short_signal = 45 <= rsi[i] <= 60
        
        # Determine target signal
        target_signal = 0.0
        
        # Long entry: HTF bullish + LTF bullish + RSI pullback
        if htf_trend == 1 and ltf_trend == 1 and rsi_long_signal:
            target_signal = BASE_SIZE
        
        # Short entry: HTF bearish + LTF bearish + RSI rally
        elif htf_trend == -1 and ltf_trend == -1 and rsi_short_signal:
            target_signal = -BASE_SIZE
        
        # Stoploss logic - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
        else:
            # Apply signal change
            if target_signal != 0.0:
                # Check if this is a reversal or new entry
                if position_side == 0:
                    # New entry
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
                elif np.sign(target_signal) == position_side:
                    # Same direction - maintain or scale
                    signals[i] = target_signal
                else:
                    # Reversal - close old, open new
                    signals[i] = target_signal
                    position_side = 1 if target_signal > 0 else -1
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
                    entry_price = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = BASE_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals