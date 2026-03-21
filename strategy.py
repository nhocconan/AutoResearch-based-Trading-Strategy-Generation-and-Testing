#!/usr/bin/env python3
"""
Experiment #391: 15m Mean Reversion + 4h HMA Trend + 1h RSI Pullback + Volume Filter
Hypothesis: 15m is too noisy for pure trend-following (see #379, #385 failures with Sharpe=-4 to -6).
Instead, use 15m for mean-reversion entries (RSI extremes) but ONLY when aligned with 4h trend.
4h HMA provides strong trend bias. 1h RSI confirms pullback within trend. Volume filter avoids
false breakouts. ATR(14) stoploss at 2.5x protects capital. Position size 0.25 discrete.
Timeframe: 15m (REQUIRED), HTF: 4h for trend, 1h for RSI confirmation via mtf_data helper.
Key insight: Mean-reversion entries WITHIN trend direction = higher win rate than pure trend-follow on 15m.
Multiple OR conditions ensure minimum trade frequency (critical - many 15m strategies got 0 trades).
Target: Beat Sharpe=0.499 (current best mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_mr_4h_hma_1h_rsi_vol_atr_v1"
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
    """Calculate Hull Moving Average for faster trend response with less lag."""
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    return vol_ratio

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = np.where(std > 0, (close - sma) / std, 0.0)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    zscore_15m = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 15m HMA for short-term trend
    hma_15m_fast = calculate_hma(close, 8)
    hma_15m_slow = calculate_hma(close, 21)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi_15m[i]) or np.isnan(zscore_15m[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_15m_fast[i]) or np.isnan(hma_15m_slow[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (STRONG filter)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i])
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 1h RSI confirmation (pullback within trend)
        rsi_1h_valid = not np.isnan(rsi_1h_aligned[i])
        rsi_1h_bullish_pullback = rsi_1h_valid and rsi_1h_aligned[i] > 40 and rsi_1h_aligned[i] < 65
        rsi_1h_bearish_pullback = rsi_1h_valid and rsi_1h_aligned[i] > 35 and rsi_1h_aligned[i] < 60
        
        # 15m RSI mean reversion signals
        rsi_15m_oversold = rsi_15m[i] < 35
        rsi_15m_overbought = rsi_15m[i] > 65
        rsi_15m_extreme_oversold = rsi_15m[i] < 25
        rsi_15m_extreme_overbought = rsi_15m[i] > 75
        
        # Z-score mean reversion
        zscore_extreme_low = zscore_15m[i] < -1.5
        zscore_extreme_high = zscore_15m[i] > 1.5
        
        # Volume confirmation
        volume_ok = vol_ratio[i] > 0.8  # Not extremely low volume
        
        # HMA crossover on 15m
        hma_cross_long = hma_15m_fast[i] > hma_15m_slow[i] and hma_15m_fast[i-1] <= hma_15m_slow[i-1]
        hma_cross_short = hma_15m_fast[i] < hma_15m_slow[i] and hma_15m_fast[i-1] >= hma_15m_slow[i-1]
        hma_15m_bullish = hma_15m_fast[i] > hma_15m_slow[i]
        hma_15m_bearish = hma_15m_fast[i] < hma_15m_slow[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple OR conditions to ensure trade frequency) ===
        # Primary: 4h bullish + 15m RSI oversold + 1h RSI pullback OK + volume OK
        if trend_4h_bullish and rsi_15m_oversold and rsi_1h_bullish_pullback and volume_ok:
            new_signal = SIZE_ENTRY
        # Secondary: 4h bullish + 15m RSI extreme oversold (stronger signal)
        elif trend_4h_bullish and rsi_15m_extreme_oversold:
            new_signal = SIZE_ENTRY
        # Tertiary: 4h bullish + Z-score extreme low + volume OK
        elif trend_4h_bullish and zscore_extreme_low and volume_ok:
            new_signal = SIZE_ENTRY
        # Quaternary: 4h bullish + 15m HMA cross long + RSI OK
        elif trend_4h_bullish and hma_cross_long and rsi_15m[i] > 35 and rsi_15m[i] < 70:
            new_signal = SIZE_ENTRY
        # Quintenary: 4h bullish + 15m HMA bullish + RSI momentum (no cross needed)
        elif trend_4h_bullish and hma_15m_bullish and rsi_15m[i] > 40 and rsi_15m[i] < 70:
            new_signal = SIZE_ENTRY
        # Sextenary: 4h neutral/valid + 15m RSI extreme + Z-score extreme (strong MR)
        elif hma_4h_valid and rsi_15m_extreme_oversold and zscore_extreme_low:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple OR conditions to ensure trade frequency) ===
        # Primary: 4h bearish + 15m RSI overbought + 1h RSI pullback OK + volume OK
        if trend_4h_bearish and rsi_15m_overbought and rsi_1h_bearish_pullback and volume_ok:
            new_signal = -SIZE_ENTRY
        # Secondary: 4h bearish + 15m RSI extreme overbought (stronger signal)
        elif trend_4h_bearish and rsi_15m_extreme_overbought:
            new_signal = -SIZE_ENTRY
        # Tertiary: 4h bearish + Z-score extreme high + volume OK
        elif trend_4h_bearish and zscore_extreme_high and volume_ok:
            new_signal = -SIZE_ENTRY
        # Quaternary: 4h bearish + 15m HMA cross short + RSI OK
        elif trend_4h_bearish and hma_cross_short and rsi_15m[i] > 30 and rsi_15m[i] < 65:
            new_signal = -SIZE_ENTRY
        # Quintenary: 4h bearish + 15m HMA bearish + RSI momentum (no cross needed)
        elif trend_4h_bearish and hma_15m_bearish and rsi_15m[i] > 30 and rsi_15m[i] < 60:
            new_signal = -SIZE_ENTRY
        # Sextenary: 4h neutral/valid + 15m RSI extreme + Z-score extreme (strong MR)
        elif hma_4h_valid and rsi_15m_extreme_overbought and zscore_extreme_high:
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