#!/usr/bin/env python3
"""
Experiment #340: 4h Dual EMA + Z-Score Mean Reversion + Daily HMA Bias + ATR Stop
Hypothesis: 4h timeframe needs hybrid trend/mean-reversion approach for bear/range markets (2025+).
Dual EMA(8/21) crossover provides trend direction, Z-score(20) filters overextended entries
(mean-reversion when |Z|>1.5, trend-follow when |Z|<1.0). Daily HMA(21) gives macro bias.
Unlike pure Supertrend/Donchian which failed on 4h, this adapts to regime changes.
Loose RSI filter (35-65) ensures sufficient trade frequency (>10 trades/symbol).
ATR(14) stoploss at 2.5x ensures risk control. Target: Beat Sharpe=0.499 with 40-80 trades/year.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Key insight: Hybrid approach works better in mixed bull/bear/range markets than pure trend-follow.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_ema_zscore_daily_hma_atr_v1"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean-reversion detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    zscore = calculate_zscore(close, 20)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(ema_fast[i]) or np.isnan(zscore[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Daily macro trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # EMA trend direction
        ema_bullish = ema_fast[i] > ema_slow[i]
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # EMA crossover signals
        ema_cross_long = ema_fast[i-1] <= ema_slow[i-1] and ema_fast[i] > ema_slow[i]
        ema_cross_short = ema_fast[i-1] >= ema_slow[i-1] and ema_fast[i] < ema_slow[i]
        
        # Z-score regime detection
        z_overbought = zscore[i] > 1.5  # Mean-reversion short candidate
        z_oversold = zscore[i] < -1.5  # Mean-reversion long candidate
        z_neutral = np.abs(zscore[i]) < 1.0  # Trend-following zone
        
        # RSI filter (LOOSE for trade frequency)
        rsi_ok_long = rsi[i] > 35  # Not extremely oversold
        rsi_ok_short = rsi[i] < 65  # Not extremely overbought
        rsi_strong_long = rsi[i] > 45
        rsi_strong_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: EMA crossover + Daily bullish + RSI ok + Z-score not overbought
        if ema_cross_long and daily_bullish and rsi_ok_long and not z_overbought:
            new_signal = SIZE_ENTRY
        # Secondary: EMA bullish + Z-score oversold (mean-reversion long)
        elif ema_bullish and z_oversold and daily_bullish and rsi_strong_long:
            new_signal = SIZE_ENTRY
        # Tertiary: EMA bullish + Z-score neutral + RSI strong (trend-follow)
        elif ema_bullish and z_neutral and daily_bullish and rsi_strong_long:
            new_signal = SIZE_ENTRY
        # Quaternary: EMA crossover without daily filter (momentum only)
        elif ema_cross_long and rsi[i] > 40 and not z_overbought:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: EMA crossover + Daily bearish + RSI ok + Z-score not oversold
        if ema_cross_short and daily_bearish and rsi_ok_short and not z_oversold:
            new_signal = -SIZE_ENTRY
        # Secondary: EMA bearish + Z-score overbought (mean-reversion short)
        elif ema_bearish and z_overbought and daily_bearish and rsi_strong_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: EMA bearish + Z-score neutral + RSI strong (trend-follow)
        elif ema_bearish and z_neutral and daily_bearish and rsi_strong_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: EMA crossover without daily filter (momentum only)
        elif ema_cross_short and rsi[i] < 60 and not z_oversold:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals