#!/usr/bin/env python3
"""
Experiment #338: 30m Keltner Breakout + 4h HMA Trend + RSI Momentum + ATR Stop
Hypothesis: 30m timeframe with Keltner Channel breakouts provides cleaner signals than Donchian.
Keltner (EMA20 + 2*ATR) adapts to volatility better than fixed Donchian periods.
4h HMA(21) provides macro trend bias to avoid counter-trend breakouts.
RSI(14) momentum filter (50-60 range for longs, 40-50 for shorts) ensures entry quality.
This combines volatility breakout (proven in exp#329) with adaptive channels and HTF filter.
Timeframe: 30m (REQUIRED), HTF: 4h for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with 30-60 trades/year, symmetric long/short logic.
Key insight: Keltner breakouts + 4h HMA filter = fewer false breakouts, better win rate in both bull/bear.
Position sizing: 0.25 entry, 0.125 half-profit, 2.5*ATR trailing stop.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_keltner_4h_hma_rsi_momentum_atr_v1"
timeframe = "30m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_keltner(high, low, close, ema_period=20, atr_period=14, atr_mult=2.0):
    """Calculate Keltner Channel bands."""
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    return upper, lower, ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    keltner_upper, keltner_lower, keltner_ema = calculate_keltner(high, low, close, 20, 14, 2.0)
    
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
    
    for i in range(250, n):  # Start after 250 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(keltner_upper[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        hma_valid = not np.isnan(hma_4h_aligned[i])
        four_h_bullish = hma_valid and close[i] > hma_4h_aligned[i]
        four_h_bearish = hma_valid and close[i] < hma_4h_aligned[i]
        
        # Keltner breakout signals (price closes outside channel)
        breakout_long = close[i] > keltner_upper[i-1] and close[i-1] <= keltner_upper[i-1]
        breakout_short = close[i] < keltner_lower[i-1] and close[i-1] >= keltner_lower[i-1]
        
        # Keltner trend state (price outside channel)
        above_upper = close[i] > keltner_upper[i-1]
        below_lower = close[i] < keltner_lower[i-1]
        
        # RSI momentum filter (moderate values, not extremes)
        rsi_ok_long = rsi[i] > 45 and rsi[i] < 75  # Not overbought
        rsi_ok_short = rsi[i] < 55 and rsi[i] > 25  # Not oversold
        
        # Strong momentum confirmation
        rsi_strong_long = rsi[i] > 50 and rsi[i] < 70
        rsi_strong_short = rsi[i] < 50 and rsi[i] > 30
        
        # EMA position relative to Keltner (trend confirmation)
        ema_above_mid = keltner_ema[i] > (keltner_upper[i] + keltner_lower[i]) / 2
        ema_below_mid = keltner_ema[i] < (keltner_upper[i] + keltner_lower[i]) / 2
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: Keltner breakout + 4h bullish + RSI ok
        if breakout_long and four_h_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: Price above upper + 4h bullish + RSI strong
        elif above_upper and four_h_bullish and rsi_strong_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Breakout with EMA confirmation (no 4h filter for momentum)
        elif breakout_long and ema_above_mid and rsi[i] > 55:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: Keltner breakout + 4h bearish + RSI ok
        if breakout_short and four_h_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: Price below lower + 4h bearish + RSI strong
        elif below_lower and four_h_bearish and rsi_strong_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Breakout with EMA confirmation (no 4h filter for momentum)
        elif breakout_short and ema_below_mid and rsi[i] < 45:
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