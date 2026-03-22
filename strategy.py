#!/usr/bin/env python3
"""
Experiment #004: 4h RSI Mean Reversion + 1d HMA Trend Filter + ATR Stop
Hypothesis: RSI extremes on 4h provide reliable mean reversion entries when aligned with 1d trend.
Simpler than CRSI, generates more frequent signals while maintaining quality via HTF filter.
1d HMA(21) provides higher-timeframe trend bias - only take longs above, shorts below.
ATR(14) trailing stop at 2.5*ATR limits drawdown during adverse moves.
Timeframe: 4h (REQUIRED for this experiment), HTF: 1d via mtf_data helper.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Key innovation: Looser RSI thresholds (40/60 vs 10/90) ensure sufficient trade frequency.
Works across BTC/ETH/SOL by following HTF trend direction rather than fighting it.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's formula."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Pad to match original length
    gain = np.pad(gain, (1, 0), 'constant')
    loss = np.pad(loss, (1, 0), 'constant')
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0  # No losses = RSI 100
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_4h = calculate_hma(close, 21)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - PRIMARY FILTER
        bull_trend = close[i] > hma_1d_aligned[i]
        bear_trend = close[i] < hma_1d_aligned[i]
        
        # 4h trend confirmation
        hma_4h_bull = close[i] > hma_4h[i]
        hma_4h_bear = close[i] < hma_4h[i]
        
        # RSI signals - LOOSE thresholds for sufficient trade frequency
        rsi_oversold = rsi[i] < 40  # Long entry zone
        rsi_overbought = rsi[i] > 60  # Short entry zone
        rsi_extreme_oversold = rsi[i] < 30  # Strong long
        rsi_extreme_overbought = rsi[i] > 70  # Strong short
        
        # SMA filter for additional confirmation
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: RSI oversold + 1d bull trend + 4h bull confirmation
        if rsi_oversold and bull_trend and hma_4h_bull:
            new_signal = SIZE_MAX
        # Secondary: RSI extreme oversold + 1d bull trend (relax 4h filter)
        elif rsi_extreme_oversold and bull_trend:
            new_signal = SIZE_BASE
        # Tertiary: RSI oversold + above SMA50 + 1d bull trend
        elif rsi_oversold and above_sma50 and bull_trend:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: RSI overbought + 1d bear trend + 4h bear confirmation
        if rsi_overbought and bear_trend and hma_4h_bear:
            new_signal = -SIZE_MAX
        # Secondary: RSI extreme overbought + 1d bear trend (relax 4h filter)
        elif rsi_extreme_overbought and bear_trend:
            new_signal = -SIZE_BASE
        # Tertiary: RSI overbought + below SMA50 + 1d bear trend
        elif rsi_overbought and below_sma50 and bear_trend:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals