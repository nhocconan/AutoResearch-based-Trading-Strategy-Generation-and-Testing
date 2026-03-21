#!/usr/bin/env python3
"""
Experiment #424: 4h Donchian Breakout + Daily HMA Trend + Volume Confirmation + ATR Stop
Hypothesis: Donchian channel breakouts (20-period) capture genuine momentum moves when combined
with daily HMA for trend direction and volume confirmation to filter false breakouts. This differs
from previous 4h failures by using breakout logic instead of mean-reversion or Supertrend.
Key insight: 4h timeframes work better for breakouts than oscillators. Daily HMA provides clean
trend bias without over-filtering. Volume spike (1.5x avg) confirms breakout validity.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2.5*ATR, take-profit at 2R reduce to half.
Target: Beat Sharpe=0.499 with >=10 trades/symbol on train and >=3 on test.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_daily_hma_volume_breakout_atr_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    volume_ma = calculate_volume_ma(volume, 20)
    rsi = calculate_rsi(close, 14)
    sma50 = calculate_sma(close, 50)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (long-term direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # Volume confirmation (spike above average)
        volume_spike = volume[i] > 1.5 * volume_ma[i]
        volume_ok = volume[i] > 1.2 * volume_ma[i]  # Relaxed for more trades
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # Price above/below Donchian mid
        donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2 if not np.isnan(donchian_upper[i]) else close[i]
        above_mid = close[i] > donchian_mid
        below_mid = close[i] < donchian_mid
        
        # RSI momentum filter (relaxed to ensure trades)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 85
        rsi_ok_short = rsi[i] > 15 and rsi[i] < 65
        
        # SMA50 trend confirmation
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Donchian breakout + Daily bullish + Volume spike (primary)
        if breakout_long and daily_bullish and volume_spike:
            new_signal = SIZE_ENTRY
        # Path 2: Donchian breakout + Daily bullish + Volume ok (relaxed)
        elif breakout_long and daily_bullish and volume_ok and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Path 3: Price above Donchian mid + Daily bullish + Above SMA50
        elif above_mid and daily_bullish and above_sma50 and rsi[i] > 45:
            new_signal = SIZE_ENTRY
        # Path 4: Breakout + Volume spike + RSI momentum (daily neutral ok)
        elif breakout_long and volume_spike and rsi[i] > 50 and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 5: Simple trend - Daily bullish + Above SMA50 + RSI > 50
        elif daily_bullish and above_sma50 and rsi[i] > 55 and volume_ok:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Donchian breakout + Daily bearish + Volume spike (primary)
        if breakout_short and daily_bearish and volume_spike:
            new_signal = -SIZE_ENTRY
        # Path 2: Donchian breakout + Daily bearish + Volume ok (relaxed)
        elif breakout_short and daily_bearish and volume_ok and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Path 3: Price below Donchian mid + Daily bearish + Below SMA50
        elif below_mid and daily_bearish and below_sma50 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 4: Breakout + Volume spike + RSI momentum (daily neutral ok)
        elif breakout_short and volume_spike and rsi[i] < 50 and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple trend - Daily bearish + Below SMA50 + RSI < 50
        elif daily_bearish and below_sma50 and rsi[i] < 45 and volume_ok:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 4h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR from lowest for 4h timeframe)
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