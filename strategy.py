#!/usr/bin/env python3
"""
Experiment #423: 1h Multi-Path Trend Following with 4h HMA Bias + Volume Confirmation
Hypothesis: Previous 1h strategies failed due to too many conflicting filters. This strategy
uses multiple independent entry paths (any one can trigger) to ensure >=10 trades/symbol.
Key features: 4h HMA for trend direction (proven in best strategies), relaxed RSI thresholds
(30-70 wide range), volume surge confirmation (1.5x average), ATR stoploss at 2.0*ATR.
Position size: 0.25 entry, 0.125 half (discrete levels to minimize fee churn).
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper (get_htf_data called ONCE before loop).
Why this might work: Simpler logic than #411/#417 which failed with Sharpe -2.666/-2.055.
Multiple entry paths ensure trades even if some conditions don't align perfectly.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_trend_4h_hma_volume_rsi_multipath_atr_v1"
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
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_sma + 1e-10)
    return vol_ratio

def calculate_momentum(close, period=10):
    """Calculate rate of change momentum."""
    roc = np.zeros(len(close))
    roc[:] = np.nan
    for i in range(period, len(close)):
        if close[i-period] != 0:
            roc[i] = (close[i] - close[i-period]) / close[i-period] * 100
    return roc

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
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma50 = calculate_sma(close, 50)
    sma200 = calculate_sma(close, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)
    momentum = calculate_momentum(close, 10)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (primary trend filter)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # SMA50 trend confirmation
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        # Volume confirmation (surge > 1.5x average)
        volume_surge = vol_ratio[i] > 1.5
        
        # RSI levels (relaxed for more trades)
        rsi_neutral_long = rsi[i] > 35 and rsi[i] < 70
        rsi_neutral_short = rsi[i] > 30 and rsi[i] < 65
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 75
        rsi_momentum_short = rsi[i] > 25 and rsi[i] < 55
        
        # Momentum confirmation
        mom_positive = momentum[i] > 0 if not np.isnan(momentum[i]) else False
        mom_negative = momentum[i] < 0 if not np.isnan(momentum[i]) else False
        
        # Price action: higher high / lower low
        hh = close[i] > close[i-5] if i >= 5 else False
        ll = close[i] < close[i-5] if i >= 5 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple independent paths) ===
        # Path 1: 4h trend + RSI neutral + volume surge (primary)
        if trend_bullish and rsi_neutral_long and volume_surge:
            new_signal = SIZE_ENTRY
        # Path 2: 4h trend + above SMA50 + RSI momentum
        elif trend_bullish and above_sma50 and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Path 3: Above SMA50 + volume surge + momentum positive
        elif above_sma50 and volume_surge and mom_positive and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 4: 4h trend + higher high + RSI ok
        elif trend_bullish and hh and rsi[i] > 40 and rsi[i] < 75:
            new_signal = SIZE_ENTRY
        # Path 5: Simple - above SMA50 + above SMA200 + RSI > 50
        elif above_sma50 and close[i] > sma200[i] and rsi[i] > 50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple independent paths) ===
        # Path 1: 4h trend + RSI neutral + volume surge (primary)
        if trend_bearish and rsi_neutral_short and volume_surge:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h trend + below SMA50 + RSI momentum
        elif trend_bearish and below_sma50 and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Path 3: Below SMA50 + volume surge + momentum negative
        elif below_sma50 and volume_surge and mom_negative and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h trend + lower low + RSI ok
        elif trend_bearish and ll and rsi[i] > 25 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple - below SMA50 + below SMA200 + RSI < 50
        elif below_sma50 and close[i] < sma200[i] and rsi[i] < 50:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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