#!/usr/bin/env python3
"""
Experiment #435: 1h Trend-Pullback with 12h HMA Bias + Volatility Regime Filter

Hypothesis: 1h timeframe failed previously due to noise and whipsaw. Key fixes:
1. Use 12h HMA (not 4h) for STRONGER trend bias - 12h aligns better with daily cycles
2. Add ATR volatility filter - avoid trades when vol < median (dead markets kill Sharpe)
3. RSI pullback in direction of HTF trend (proven in 12h best strategy)
4. Multiple entry paths but with COMMON filter (12h trend MUST agree)
5. Relaxed RSI (35-65 range) to ensure >=10 trades per symbol

Why 12h not 4h: 4h has 6 bars/day, 12h has 2 bars/day. 12h smoother, less false trend flips.
The best strategy (Sharpe=0.499) uses 12h primary - we adapt to 1h entries with 12h bias.

Timeframe: 1h (REQUIRED for this experiment)
HTF: 12h for trend bias via mtf_data helper
Position size: 0.30 discrete, stoploss 2.5*ATR
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_12h_hma_vol_regime_atr_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l > 0)
    rsi = 100 - 100 / (1 + rs)
    return np.clip(rsi, 0, 100)

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram for momentum."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return (macd_line - signal_line).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for trend bias
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align 12h to 1h (Rule 2 - auto shift(1) for completed bars)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_hist = calculate_macd(close, 12, 26, 9)
    sma50 = calculate_sma(close, 50)
    
    # Calculate volatility regime (ATR percentile)
    atr_series = pd.Series(atr)
    atr_median = atr_series.rolling(window=100, min_periods=50).median().values
    vol_ok = atr > 0.5 * atr_median  # Only trade when vol > 50% of recent median
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_hist[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ok[i]):
            signals[i] = 0.0
            continue
        
        # 12h trend bias (PRIMARY FILTER - must agree for all entries)
        trend_bullish = close[i] > hma_12h_aligned[i]
        trend_bearish = close[i] < hma_12h_aligned[i]
        
        # Volatility filter - avoid dead markets
        vol_active = vol_ok[i]
        
        # RSI pullback conditions (RELAXED for more trades)
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 55 and rsi[i] > rsi[i-1]
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 65 and rsi[i] < rsi[i-1]
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # MACD momentum
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        macd_improving_long = macd_hist[i] > macd_hist[i-1]
        macd_improving_short = macd_hist[i] < macd_hist[i-1]
        
        # Price vs SMA50
        above_sma = close[i] > sma50[i]
        below_sma = close[i] < sma50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths, all require 12h bullish) ===
        # Path 1: 12h bullish + RSI pullback + MACD improving
        if trend_bullish and vol_active and rsi_pullback_long and macd_improving_long:
            new_signal = SIZE_ENTRY
        # Path 2: 12h bullish + RSI oversold + MACD bullish
        elif trend_bullish and vol_active and rsi_oversold and macd_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: 12h bullish + Above SMA50 + RSI 40-60 + MACD positive
        elif trend_bullish and vol_active and above_sma and rsi[i] > 40 and rsi[i] < 60 and macd_bullish:
            new_signal = SIZE_ENTRY
        # Path 4: 12h bullish + RSI crossing up through 45
        elif trend_bullish and vol_active and rsi[i] > 45 and rsi[i-1] <= 45:
            new_signal = SIZE_ENTRY
        # Path 5: 12h bullish + MACD cross up (hist was neg, now pos)
        elif trend_bullish and vol_active and macd_hist[i] > 0 and macd_hist[i-1] <= 0:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths, all require 12h bearish) ===
        # Path 1: 12h bearish + RSI pullback + MACD improving
        if trend_bearish and vol_active and rsi_pullback_short and macd_improving_short:
            new_signal = -SIZE_ENTRY
        # Path 2: 12h bearish + RSI overbought + MACD bearish
        elif trend_bearish and vol_active and rsi_overbought and macd_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: 12h bearish + Below SMA50 + RSI 40-60 + MACD negative
        elif trend_bearish and vol_active and below_sma and rsi[i] > 40 and rsi[i] < 60 and macd_bearish:
            new_signal = -SIZE_ENTRY
        # Path 4: 12h bearish + RSI crossing down through 55
        elif trend_bearish and vol_active and rsi[i] < 55 and rsi[i-1] >= 55:
            new_signal = -SIZE_ENTRY
        # Path 5: 12h bearish + MACD cross down (hist was pos, now neg)
        elif trend_bearish and vol_active and macd_hist[i] < 0 and macd_hist[i-1] >= 0:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals