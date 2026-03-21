#!/usr/bin/env python3
"""
Experiment #068: 30m KAMA Adaptive Trend with 4h/1d HMA Dual Filter + Volume Confirmation
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than EMA,
reducing whipsaws in choppy markets. Combined with dual HTF HMA filters (4h for medium-term,
1d for long-term bias), this should catch sustained trends while filtering noise.
Volume spike confirmation (>1.5x 20-bar avg) reduces false breakouts.
30m timeframe provides good balance between trade frequency and signal quality.
Position sizing: 0.28 entry, 0.14 half at 1.5R profit, stoploss at 2.5*ATR trailing.
Entry conditions simplified to ensure 10+ trades per symbol (learning from 0-trade failures).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_1d_hma_volume_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast in trending markets, slow in choppy.
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    close_s = pd.Series(close)
    
    # Net change over period
    net_change = close_s.diff(period).abs()
    
    # Sum of absolute changes over period
    abs_changes = close_s.diff().abs()
    sum_abs_changes = abs_changes.rolling(window=period, min_periods=period).sum()
    
    # Efficiency Ratio (0 to 1)
    er = net_change / sum_abs_changes.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period] = close[period]  # Initialize with price
    
    for i in range(period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
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

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above threshold * average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    spike = volume > (threshold * vol_avg.values)
    return spike

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
    kama_10 = calculate_kama(close, period=10, fast=2, slow=30)
    kama_30 = calculate_kama(close, period=30, fast=2, slow=30)
    volume_spike = calculate_volume_spike(volume, 20, 1.5)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # HTF trend filters
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA crossover signals (adaptive trend)
        kama_cross_long = kama_10[i] > kama_30[i] and (i > 0 and kama_10[i-1] <= kama_30[i-1])
        kama_cross_short = kama_10[i] < kama_30[i] and (i > 0 and kama_10[i-1] >= kama_30[i-1])
        
        # KAMA trend state
        kama_trend_long = kama_10[i] > kama_30[i]
        kama_trend_short = kama_10[i] < kama_30[i]
        
        # KAMA slope confirmation
        kama_slope_long = kama_10[i] > kama_10[i-1] if i > 0 else False
        kama_slope_short = kama_10[i] < kama_10[i-1] if i > 0 else False
        
        # RSI filter (avoid extremes)
        rsi_ok_long = rsi[i] < 70
        rsi_ok_short = rsi[i] > 30
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        new_signal = 0.0
        
        # LONG ENTRY conditions (simplified to ensure trades)
        # Primary: KAMA cross + 4h bullish + volume spike OR RSI ok
        if kama_cross_long and hma_4h_bullish and (vol_confirmed or rsi_ok_long):
            new_signal = SIZE_ENTRY
        # Secondary: KAMA trend + 4h bullish + 1d bullish (strong trend alignment)
        elif kama_trend_long and hma_4h_bullish and hma_1d_bullish and kama_slope_long:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA cross + 1d bullish (catch early entries)
        elif kama_cross_long and hma_1d_bullish:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Primary: KAMA cross + 4h bearish + volume spike OR RSI ok
        if kama_cross_short and hma_4h_bearish and (vol_confirmed or rsi_ok_short):
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA trend + 4h bearish + 1d bearish (strong trend alignment)
        elif kama_trend_short and hma_4h_bearish and hma_1d_bearish and kama_slope_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA cross + 1d bearish (catch early entries)
        elif kama_cross_short and hma_1d_bearish:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 1.5 * risk:
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
                # Take profit at 1.5R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 1.5 * risk:
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