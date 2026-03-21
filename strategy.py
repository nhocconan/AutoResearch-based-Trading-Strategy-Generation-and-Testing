#!/usr/bin/env python3
"""
Experiment #006: Daily EMA Trend + Weekly Regime + ATR Stop
Hypothesis: Daily timeframe captures major crypto swings with minimal noise.
Weekly HTF provides major bull/bear regime filter (avoid counter-trend trades in crashes).
Daily EMA crossover (12/26) gives entry signals with RSI confirmation.
Relaxed entry conditions to ensure ≥10 trades/symbol on train data.
ATR-based stoploss (3x) protects against crashes like 2022.
Position sizing at 0.35 with discrete levels to minimize fee churn.

Key differences from #005:
- Primary TF is 1d (not 12h) = fewer but higher quality trades
- Weekly regime filter (not daily) = better major trend alignment
- Simpler entry logic = more reliable signals, more trades
- Wider ATR stop (3x vs 2.5x) for daily timeframe volatility
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_ema_weekly_regime_atr_v1"
timeframe = "1d"
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

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load weekly HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    ema_1w = calculate_ema(df_1w['close'].values, 21)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily indicators
    atr = calculate_atr(high, low, close, 14)
    ema_fast = calculate_ema(close, 12)
    ema_slow = calculate_ema(close, 26)
    rsi = calculate_rsi(close, 14)
    
    # Price momentum for confirmation
    roc_5 = np.zeros(n)
    roc_5[5:] = (close[5:] - close[:-5]) / close[:-5] * 100
    
    signals = np.zeros(n)
    SIZE = 0.35
    HALF_SIZE = 0.18
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    max_close = np.zeros(n)
    min_close = np.zeros(n)
    
    for i in range(50, n):
        # Weekly trend filter (major regime) - soft filter
        weekly_bullish = ema_1w_aligned[i] > 0 and close[i] > ema_1w_aligned[i]
        weekly_bearish = ema_1w_aligned[i] > 0 and close[i] < ema_1w_aligned[i]
        
        # Daily EMA crossover
        ema_cross_long = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_short = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # EMA trend
        ema_trend_long = ema_fast[i] > ema_slow[i]
        ema_trend_short = ema_fast[i] < ema_slow[i]
        
        # RSI filter (relaxed for more trades)
        rsi_ok_long = rsi[i] > 25 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 75
        
        # Momentum confirmation
        mom_long = roc_5[i] > -5  # Not strongly negative
        mom_short = roc_5[i] < 5  # Not strongly positive
        
        # Entry logic - relaxed conditions to ensure trades
        new_signal = 0.0
        
        # Long entry: weekly bullish + EMA trend + RSI ok
        if weekly_bullish and ema_trend_long and rsi_ok_long:
            new_signal = SIZE
        # Long on EMA crossover with weekly support
        elif ema_cross_long and (weekly_bullish or rsi[i] > 40):
            new_signal = SIZE
        # Long on RSI oversold bounce with EMA support
        elif rsi[i] < 35 and ema_trend_long and mom_long:
            new_signal = SIZE
        
        # Short entry: weekly bearish + EMA trend + RSI ok
        elif weekly_bearish and ema_trend_short and rsi_ok_short:
            new_signal = -SIZE
        # Short on EMA crossover with weekly resistance
        elif ema_cross_short and (weekly_bearish or rsi[i] < 60):
            new_signal = -SIZE
        # Short on RSI overbought drop with EMA resistance
        elif rsi[i] > 65 and ema_trend_short and mom_short:
            new_signal = -SIZE
        
        # Stoploss logic - ATR based (3x for daily timeframe)
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 3.0 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs
            else:
                max_close[i] = max(max_close[i-1] if i > 0 else 0, close[i])
                trail_stop = max_close[i] - 3.0 * atr[i]
                if close[i] < trail_stop and trail_stop > entry_price[i-1]:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] > entry_price[i-1] + 3.0 * atr[i] and new_signal == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 3.0 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts
            else:
                min_close[i] = min(min_close[i-1] if i > 0 else 999999, close[i])
                trail_stop = min_close[i] + 3.0 * atr[i]
                if close[i] > trail_stop and trail_stop < entry_price[i-1]:
                    new_signal = 0.0
                # Take partial profit at 3R
                elif close[i] < entry_price[i-1] - 3.0 * atr[i] and new_signal == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            max_close[i] = close[i]
            min_close[i] = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                max_close[i] = close[i]
                min_close[i] = close[i]
            else:
                max_close[i] = max_close[i-1] if i > 0 else close[i]
                min_close[i] = min_close[i-1] if i > 0 else close[i]
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            max_close[i] = max_close[i-1] if i > 0 else 0
            min_close[i] = min_close[i-1] if i > 0 else 0
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals