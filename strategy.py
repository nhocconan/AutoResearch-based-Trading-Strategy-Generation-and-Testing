#!/usr/bin/env python3
"""
EXPERIMENT #002 - MACD-RSI Ensemble with 4h Trend Filter (30m)
=================================================================
Hypothesis: 30m MACD histogram momentum combined with RSI pullback entries,
filtered by 4h HMA trend direction, captures medium-term swings while avoiding
counter-trend trades. ATR trailing stop protects against reversals.

Key features:
- Primary TF: 30m (balances noise reduction vs trade frequency)
- HTF filter: 4h HMA(21) for trend direction
- Entry: MACD histogram turning + RSI(14) pullback to 40-60 zone
- Filter: Price above/below 4h HMA confirms trend
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels to minimize fee churn

Why 30m: Less noisy than 15m, more trades than 1h. Good for swing captures.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "macd_rsi_4h_trend_30m_v1"
timeframe = "30m"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average for trend direction"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values


def calculate_rsi(close, period=14):
    """Calculate RSI oscillator"""
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
    """Calculate Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_sma(close, period):
    """Calculate Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # Calculate 30m indicators (all before loop for performance)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30  # Entry position size (30% of capital)
    SIZE_HALF = 0.15   # Half position for take profit
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    take_profit_hit = False
    
    min_period = 220  # Wait for SMA200 and all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(sma_200[i]) or 
            atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h Trend filter (HTF alignment ensures no look-ahead)
        trend_4h = 1 if close[i] > hma_4h_aligned[i] else -1
        
        # 30m trend filter (price above/below SMA200)
        trend_30m = 1 if close[i] > sma_200[i] else -1
        
        # MACD histogram momentum (turning points)
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]
        
        # RSI pullback zone (not at extremes, allows entry on pullback)
        rsi_long_ok = 40 < rsi[i] < 65
        rsi_short_ok = 35 < rsi[i] < 60
        
        # Determine target signal based on ensemble
        target_signal = 0.0
        
        # Long entry: 4h trend up + 30m trend up + MACD bullish + RSI in zone
        if trend_4h == 1 and trend_30m == 1 and macd_bullish and rsi_long_ok:
            target_signal = SIZE_ENTRY
        
        # Short entry: 4h trend down + 30m trend down + MACD bearish + RSI in zone
        elif trend_4h == -1 and trend_30m == -1 and macd_bearish and rsi_short_ok:
            target_signal = -SIZE_ENTRY
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R profit = entry + 2*ATR at entry)
                profit_target = entry_price + 2.0 * atr[i]
                if not take_profit_hit and close[i] >= profit_target:
                    take_profit_triggered = True
                    take_profit_hit = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R profit = entry - 2*ATR at entry)
                profit_target = entry_price - 2.0 * atr[i]
                if not take_profit_hit and close[i] <= profit_target:
                    take_profit_triggered = True
                    take_profit_hit = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            take_profit_hit = False
        elif take_profit_triggered:
            # Reduce to half position at 2R profit
            signals[i] = SIZE_HALF * position_side
            # Trail stop at 1R now
            # Position remains open but reduced
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                entry_price = close[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                take_profit_hit = False
            elif position_side != 0:
                # Maintain existing position (don't flip unless stoploss)
                # Only allow signal change if same direction or flat
                if target_signal == 0.0:
                    # Exit signal
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    take_profit_hit = False
                elif np.sign(target_signal) == position_side:
                    # Same direction - maintain
                    signals[i] = SIZE_ENTRY * position_side
                # else: ignore opposite signal (no flipping without exit)
            else:
                signals[i] = 0.0
    
    return signals