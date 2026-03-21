#!/usr/bin/env python3
"""
Experiment #217: 15m Multi-Timeframe Trend Pullback with Volume Confirmation
Hypothesis: 15m timeframe is noisy, so we need strong HTF filters. Use 4h HMA for 
major trend direction, 1h RSI for pullback timing, and volume confirmation to reduce 
false breakouts. Entry: Long when price > 4h HMA + 1h RSI pulls back to 40-50 + volume 
> 1.5x average. Short when price < 4h HMA + 1h RSI rallies to 50-60 + volume spike. 
Stoploss: 2.5*ATR trailing. Position sizing: 0.25 entry, 0.125 half at 2R profit.
This should generate more trades than 12h/1d strategies while maintaining quality.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_4h_hma_1h_rsi_volume_pullback_atr_v1"
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

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Volume ratio (current vs average)
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    vol_ratio = np.clip(vol_ratio, 0.5, 5.0)
    
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
    
    for i in range(100, n):
        # HTF trend filter (4h HMA)
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h RSI pullback zones
        rsi_pullback_long = 40 <= rsi_1h_aligned[i] <= 55
        rsi_pullback_short = 45 <= rsi_1h_aligned[i] <= 60
        
        # 15m RSI confirmation (not overbought/oversold)
        rsi_15m_ok_long = 35 <= rsi_15m[i] <= 65
        rsi_15m_ok_short = 35 <= rsi_15m[i] <= 65
        
        # Volume confirmation (above average)
        volume_confirmed = vol_ratio[i] > 1.2
        
        # Price momentum (close above/below previous)
        momentum_long = close[i] > close[i-1]
        momentum_short = close[i] < close[i-1]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Trend bullish + RSI pullback + volume + momentum
        if trend_bullish and rsi_pullback_long and rsi_15m_ok_long:
            if volume_confirmed and momentum_long:
                new_signal = SIZE_ENTRY
            elif not volume_confirmed and momentum_long and rsi_15m[i] < 50:
                # Enter on weaker volume if RSI is low enough
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Trend bearish + RSI pullback + volume + momentum
        if trend_bearish and rsi_pullback_short and rsi_15m_ok_short:
            if volume_confirmed and momentum_short:
                new_signal = -SIZE_ENTRY
            elif not volume_confirmed and momentum_short and rsi_15m[i] > 50:
                # Enter on weaker volume if RSI is high enough
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