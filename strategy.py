#!/usr/bin/env python3
"""
Experiment #176: 30m Multi-Timeframe Trend-Pullback Strategy
Hypothesis: 30m timeframe captures intraday swings while 4h HMA provides trend bias.
Simple RSI pullback entries (loosened to 35/65 for more trades) + volume confirmation.
Key insight from failures: entry conditions were TOO STRICT causing 0 trades.
This strategy loosens RSI thresholds, reduces filter count, and ensures trades happen.
Position sizing: 0.25 entry, 0.15 half-size at 2R profit. Stoploss at 2.5*ATR.
Target: Beat Sharpe=0.499 from current best while ensuring ≥10 trades per symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_volume_pullback_atr_v1"
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

def calculate_sma(values, period=20):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    vol_sma = calculate_sma(volume, 20)
    close_sma = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters (loosened - just check slope direction)
        hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-5] if i > 5 else 0
        hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-5] if i > 5 else 0
        
        trend_4h_bullish = hma_4h_slope > 0
        trend_4h_bearish = hma_4h_slope < 0
        trend_1d_bullish = hma_1d_slope > 0
        trend_1d_bearish = hma_1d_slope < 0
        
        # 30m trend
        trend_30m_bullish = hma_20[i] > hma_50[i]
        trend_30m_bearish = hma_20[i] < hma_50[i]
        
        # Price vs 200 SMA (major trend filter)
        price_above_sma200 = close[i] > close_sma[i]
        price_below_sma200 = close[i] < close_sma[i]
        
        # Volume confirmation
        volume_above_avg = volume[i] > vol_sma[i] * 0.8  # Loosened from 1.0
        
        # RSI signals (LOOSENED for more trades - was 30/70, now 35/65)
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else False
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else False
        rsi_neutral = 35 <= rsi[i] <= 65
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure trades) ===
        # Path 1: Trend pullback (4h bullish + RSI oversold)
        if trend_4h_bullish and rsi_oversold and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # Path 2: Breakout continuation (30m bullish + volume + RSI rising)
        elif trend_30m_bullish and volume_above_avg and rsi[i] > 50 and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # Path 3: Major trend alignment (1d bullish + price > SMA200 + RSI neutral rising)
        elif trend_1d_bullish and price_above_sma200 and rsi_neutral and rsi_rising:
            new_signal = SIZE_ENTRY
        
        # Path 4: Simple RSI reversal (very loose - ensures trades in ranging market)
        elif rsi[i] < 35 and rsi_rising:
            new_signal = SIZE_ENTRY * 0.8  # Smaller size for counter-trend
        
        # === SHORT ENTRIES (mirror of long) ===
        # Path 1: Trend pullback (4h bearish + RSI overbought)
        elif trend_4h_bearish and rsi_overbought and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 2: Breakdown continuation (30m bearish + volume + RSI falling)
        elif trend_30m_bearish and volume_above_avg and rsi[i] < 50 and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Major trend alignment (1d bearish + price < SMA200 + RSI neutral falling)
        elif trend_1d_bearish and price_below_sma200 and rsi_neutral and rsi_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 4: Simple RSI reversal (very loose - ensures trades in ranging market)
        elif rsi[i] > 65 and rsi_falling:
            new_signal = -SIZE_ENTRY * 0.8  # Smaller size for counter-trend
        
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