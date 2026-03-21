#!/usr/bin/env python3
"""
Experiment #355: 15m Multi-Timeframe Supertrend + RSI Pullback + 4h HMA Trend
Hypothesis: 15m timeframe captures intraday momentum while 4h HMA provides macro trend bias.
Supertrend(10,3) on 1h gives intermediate trend confirmation. RSI(14) pullback entries
in trend direction reduce whipsaws. ATR(14) stoploss at 2.5x protects capital.
Timeframe: 15m (REQUIRED), HTF: 1h Supertrend + 4h HMA via mtf_data helper.
Key insight: 3-tier MTF (4h trend + 1h momentum + 15m entry) filters false signals in ranges.
Position sizing: 0.25 entry, 0.125 half (discrete levels to minimize fee churn).
Target: Beat Sharpe=0.499 with 50-100 trades on train, 15-30 on test.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_1h_4h_hma_rsi_pullback_atr_v1"
timeframe = "15m"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
    
    return supertrend, direction

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    supertrend_1h, st_direction_1h = calculate_supertrend(
        df_1h['high'].values, 
        df_1h['low'].values, 
        df_1h['close'].values, 
        period=10, 
        multiplier=3.0
    )
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    st_direction_1h_aligned = align_htf_to_ltf(prices, df_1h, st_direction_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
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
    
    for i in range(250, n):  # Start after 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias (HMA direction)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 1h Supertrend direction (intermediate trend)
        st_1h_valid = not np.isnan(st_direction_1h_aligned[i])
        st_1h_bullish = st_1h_valid and st_direction_1h_aligned[i] > 0
        st_1h_bearish = st_1h_valid and st_direction_1h_aligned[i] < 0
        
        # 15m SMA200 filter (long-term bias)
        sma_200_valid = not np.isnan(sma_200[i])
        above_sma200 = sma_200_valid and close[i] > sma_200[i]
        below_sma200 = sma_200_valid and close[i] < sma_200[i]
        
        # RSI pullback levels (loose for 15m to ensure trades)
        rsi_oversold = rsi[i] < 45  # Pullback in uptrend
        rsi_overbought = rsi[i] > 55  # Pullback in downtrend
        rsi_extreme_long = rsi[i] < 35  # Deep oversold
        rsi_extreme_short = rsi[i] > 65  # Deep overbought
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: 4h bullish + 1h Supertrend bullish + RSI pullback
        if trend_4h_bullish and st_1h_bullish and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + above SMA200 + RSI extreme (mean reversion)
        elif trend_4h_bullish and above_sma200 and rsi_extreme_long:
            new_signal = SIZE_ENTRY
        # Tertiary: 1h Supertrend bullish + RSI extreme (momentum only - ensures trades)
        elif st_1h_bullish and rsi_extreme_long:
            new_signal = SIZE_ENTRY
        # Quaternary: Price above SMA200 + RSI recovering from oversold
        elif above_sma200 and rsi[i] > 30 and rsi[i] < 50:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: 4h bearish + 1h Supertrend bearish + RSI pullback
        if trend_4h_bearish and st_1h_bearish and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + below SMA200 + RSI extreme (mean reversion)
        elif trend_4h_bearish and below_sma200 and rsi_extreme_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: 1h Supertrend bearish + RSI extreme (momentum only - ensures trades)
        elif st_1h_bearish and rsi_extreme_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: Price below SMA200 + RSI recovering from overbought
        elif below_sma200 and rsi[i] > 50 and rsi[i] < 70:
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