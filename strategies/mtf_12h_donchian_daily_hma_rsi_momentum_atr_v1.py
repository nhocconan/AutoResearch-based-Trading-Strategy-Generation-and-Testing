#!/usr/bin/env python3
"""
Experiment #425: 12h Donchian Breakout + Daily HMA Trend + RSI Momentum + ATR Stop
Hypothesis: Donchian channel breakouts (20-period) generate more frequent signals than Fisher
Transform while still capturing trend moves. Combined with 1d HMA for trend bias and relaxed
RSI filter, this should generate >=10 trades/symbol while maintaining positive Sharpe.
Key insight from failures: Need multiple entry paths with relaxed filters to ensure trades.
Donchian breakouts occur more often than oscillator crosses, especially on 12h timeframe.
Timeframe: 12h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Position size: 0.25 discrete, stoploss 2.5*ATR for 12h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_daily_hma_rsi_momentum_atr_v1"
timeframe = "12h"
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

def calculate_donchian_channels(high, low, period=20):
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
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma50 = calculate_sma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # Donchian middle line
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (long-term direction)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 12h trend filter
        above_sma50 = close[i] > sma50[i]
        below_sma50 = close[i] < sma50[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i > 0 else False
        
        # Donchian position (price relative to channel)
        in_upper_half = close[i] > donchian_mid[i]
        in_lower_half = close[i] < donchian_mid[i]
        
        # RSI momentum (RELAXED thresholds to ensure trades)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 85
        rsi_ok_short = rsi[i] > 15 and rsi[i] < 65
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        # RSI not extreme (avoid buying top/selling bottom)
        rsi_not_overbought = rsi[i] < 75
        rsi_not_oversold = rsi[i] > 25
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Donchian breakout + Daily bullish + RSI ok (primary)
        if breakout_long and daily_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Path 2: Donchian breakout + Above SMA50 + RSI momentum
        elif breakout_long and above_sma50 and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Path 3: Price in upper Donchian + Daily bullish + RSI not overbought
        elif in_upper_half and daily_bullish and rsi_not_overbought and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Path 4: Donchian breakout + RSI > 50 (momentum confirmation)
        elif breakout_long and rsi[i] > 50 and daily_bullish:
            new_signal = SIZE_ENTRY
        # Path 5: Simple trend - price > SMA50 + Daily bullish + RSI > 45
        elif above_sma50 and daily_bullish and rsi[i] > 45 and rsi[i] < 80:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: Donchian breakout + Daily bearish + RSI ok (primary)
        if breakout_short and daily_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Path 2: Donchian breakout + Below SMA50 + RSI momentum
        elif breakout_short and below_sma50 and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Path 3: Price in lower Donchian + Daily bearish + RSI not oversold
        elif in_lower_half and daily_bearish and rsi_not_oversold and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 4: Donchian breakout + RSI < 50 (momentum confirmation)
        elif breakout_short and rsi[i] < 50 and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Path 5: Simple trend - price < SMA50 + Daily bearish + RSI < 55
        elif below_sma50 and daily_bearish and rsi[i] < 55 and rsi[i] > 20:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for 12h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR from lowest for 12h timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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