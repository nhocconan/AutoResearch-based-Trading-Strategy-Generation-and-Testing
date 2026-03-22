#!/usr/bin/env python3
"""
Experiment #013: 15m RSI Mean Reversion + 4h HMA Trend Filter + Volume Confirmation
Hypothesis: 15m timeframe captures intraday mean reversion opportunities while 4h HMA 
provides trend bias to avoid counter-trend trades. Volume confirmation reduces false 
signals. More lenient RSI thresholds (40/60 vs 20/80) ensure sufficient trade generation.

Key learnings from 12 failed experiments:
- Entry conditions must NOT be too strict (many got 0 trades = auto-reject)
- RSI thresholds of 40/60 generate more signals than 20/80
- Volume confirmation adds filter without being too restrictive
- 4h HMA trend filter prevents counter-trend disasters in strong trends
- ATR trailing stop at 2.0*ATR limits drawdown

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (get_htf_data called ONCE before loop)
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_4h_hma_volume_v1"
timeframe = "15m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    return rsi.values

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

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation."""
    volume_s = pd.Series(volume)
    vol_sma = volume_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    # Additional trend filter on 15m
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - use completed 4h bar only
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Volume confirmation (must be above average)
        volume_confirmed = volume[i] > 1.2 * vol_sma[i]
        
        # RSI signals - LENIENT thresholds to ensure trades (key lesson from failures)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_extreme_oversold = rsi[i] < 30
        rsi_extreme_overbought = rsi[i] > 70
        
        # EMA trend confirmation on 15m
        ema_bullish = close[i] > ema_50[i]
        ema_bearish = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        if rsi_extreme_oversold and bull_trend:
            # Extreme oversold + HTF bull = strong long
            new_signal = SIZE_MAX
        elif rsi_oversold and bull_trend and volume_confirmed:
            # Oversold + HTF bull + volume = standard long
            new_signal = SIZE_BASE
        elif rsi_oversold and bull_trend:
            # Oversold + HTF bull (no volume) = weaker long
            new_signal = SIZE_BASE * 0.8
        elif rsi_extreme_oversold and ema_bullish:
            # Extreme oversold + 15m EMA bull = fallback long
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        elif rsi_extreme_overbought and bear_trend:
            # Extreme overbought + HTF bear = strong short
            new_signal = -SIZE_MAX
        elif rsi_overbought and bear_trend and volume_confirmed:
            # Overbought + HTF bear + volume = standard short
            new_signal = -SIZE_BASE
        elif rsi_overbought and bear_trend:
            # Overbought + HTF bear (no volume) = weaker short
            new_signal = -SIZE_BASE * 0.8
        elif rsi_extreme_overbought and ema_bearish:
            # Extreme overbought + 15m EMA bear = fallback short
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = lowest_close + 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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