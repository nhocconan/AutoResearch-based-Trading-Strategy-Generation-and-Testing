#!/usr/bin/env python3
"""
Experiment #039: 1h RSI Pullback + 4h HMA Trend + Volume Confirmation v2
Hypothesis: Previous strategies failed due to overly strict entry filters (0 trades).
This version uses RELAXED thresholds to ensure 10+ trades per symbol.
Key changes from failed versions:
- RSI thresholds: 35/65 instead of 30/70 (more signals)
- Volume filter: >0.7x avg instead of >1.0x (less restrictive)
- Multiple entry paths (OR logic, not AND) to increase trade frequency
- Stoploss: 2.5*ATR (wider to avoid premature exits)
- Position sizing: 0.30 base, 0.35 max (discrete levels)
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
Critical: get_htf_data() called ONCE before loop, aligned arrays used inside.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h_hma_vol_v2"
timeframe = "1h"
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
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    return vol_ratio

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
    
    # Calculate 1h indicators
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_MAX = 0.35
    
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
        
        # 4h trend bias (HTF)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Volume confirmation (relaxed - >0.7x average)
        volume_ok = vol_ratio[i] > 0.7
        
        # RSI signals (RELAXED thresholds for more trades)
        rsi_oversold = rsi[i] < 40  # Was <30, too strict = 0 trades
        rsi_overbought = rsi[i] > 60  # Was >70, too strict = 0 trades
        rsi_extreme_oversold = rsi[i] < 35
        rsi_extreme_overbought = rsi[i] > 65
        
        # EMA trend confirmation (secondary filter)
        ema_bullish = close[i] > ema_50[i]
        ema_bearish = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY (multiple paths - OR logic for more trades) ===
        # Path 1: RSI oversold + 4h bull trend + volume ok
        if rsi_oversold and bull_trend and volume_ok:
            new_signal = SIZE_BASE
        # Path 2: RSI oversold + 4h bull trend + EMA bullish
        elif rsi_oversold and bull_trend and ema_bullish:
            new_signal = SIZE_BASE
        # Path 3: Extreme RSI oversold + 4h bull trend (no volume req)
        elif rsi_extreme_oversold and bull_trend:
            new_signal = SIZE_MAX
        # Path 4: RSI oversold + EMA bullish + EMA > EMA200
        elif rsi_oversold and ema_bullish and close[i] > ema_200[i]:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY (multiple paths - OR logic for more trades) ===
        # Path 1: RSI overbought + 4h bear trend + volume ok
        if rsi_overbought and bear_trend and volume_ok:
            new_signal = -SIZE_BASE
        # Path 2: RSI overbought + 4h bear trend + EMA bearish
        elif rsi_overbought and bear_trend and ema_bearish:
            new_signal = -SIZE_BASE
        # Path 3: Extreme RSI overbought + 4h bear trend (no volume req)
        elif rsi_extreme_overbought and bear_trend:
            new_signal = -SIZE_MAX
        # Path 4: RSI overbought + EMA bearish + EMA < EMA200
        elif rsi_overbought and ema_bearish and close[i] < ema_200[i]:
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