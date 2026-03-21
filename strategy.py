#!/usr/bin/env python3
"""
Experiment #251: 12h KAMA Adaptive Trend + HMA Slope with Daily/Weekly Regime Filter
Hypothesis: KAMA adapts to market noise better than EMA/HMA alone. Combined with HMA slope
for trend direction and simple RSI filter, this should generate more consistent trades than
MACD/Donchian complex entries. Key change: LOOSEN entry conditions to ensure trades generate.
Use volatility percentile to avoid extreme chop. Position sizing: 0.25 entry, 0.125 half at 2R.
Stoploss: 2.0*ATR trailing stop. Target: Beat Sharpe=0.499 with more trades and lower DD.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_hma_slope_daily_weekly_regime_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(er_period))
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = np.where(volatility > 0, change / volatility, 0)
    er = pd.Series(er).fillna(0).values
    
    fast_sc = (2 / (fast_period + 1)) ** 2
    slow_sc = (2 / (slow_period + 1)) ** 2
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

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

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope (positive = uptrend, negative = downtrend)."""
    slope = np.zeros(len(hma))
    for i in range(lookback, len(hma)):
        slope[i] = (hma[i] - hma[i-lookback]) / hma[i-lookback] * 100
    return slope

def calculate_volatility_percentile(atr, period=50):
    """Calculate ATR percentile to detect high/low volatility regimes."""
    atr_s = pd.Series(atr)
    percentile = atr_s.rolling(window=period, min_periods=period).apply(
        lambda x: np.searchsorted(np.sort(x.values), x.iloc[-1]) / len(x), raw=False
    ).values
    return np.nan_to_num(percentile, nan=0.5)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_12h = calculate_hma(close, 21)
    kama_12h = calculate_kama(close, er_period=10)
    hma_slope = calculate_hma_slope(hma_12h, lookback=5)
    vol_percentile = calculate_volatility_percentile(atr, period=50)
    
    # Track previous values
    prev_hma_slope = np.roll(hma_slope, 1)
    prev_hma_slope[0] = hma_slope[0]
    prev_kama = np.roll(kama_12h, 1)
    prev_kama[0] = kama_12h[0]
    
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
        # HTF trend filters (simple price vs HMA)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # 12h trend signals
        hma_slope_positive = hma_slope[i] > 0
        hma_slope_negative = hma_slope[i] < 0
        hma_slope_increasing = hma_slope[i] > prev_hma_slope[i]
        hma_slope_decreasing = hma_slope[i] < prev_hma_slope[i]
        
        # KAMA trend
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        kama_cross_up = prev_kama[i] >= close[i-1] and close[i] > kama_12h[i]
        kama_cross_down = prev_kama[i] <= close[i-1] and close[i] < kama_12h[i]
        
        # RSI filter (loose: 25-75 to ensure trades)
        rsi_neutral = 25 < rsi[i] < 75
        rsi_bullish = rsi[i] > 35
        rsi_bearish = rsi[i] < 65
        
        # Volatility regime (avoid extreme volatility)
        vol_normal = 0.2 < vol_percentile[i] < 0.8
        
        new_signal = 0.0
        
        # === LONG ENTRY (loose conditions to ensure trades) ===
        # KAMA cross up with daily trend
        if kama_cross_up:
            if daily_bullish and rsi_bullish:
                new_signal = SIZE_ENTRY
            elif weekly_bullish and hma_slope_positive:
                new_signal = SIZE_ENTRY
        
        # HMA slope turn positive with KAMA confirmation
        elif hma_slope_positive and hma_slope_increasing:
            if kama_bullish and (daily_bullish or weekly_bullish):
                new_signal = SIZE_ENTRY
        
        # Price above both KAMA and HMA in uptrend
        elif kama_bullish and hma_slope_positive:
            if rsi_bullish and vol_normal:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY (loose conditions to ensure trades) ===
        # KAMA cross down with daily trend
        if kama_cross_down:
            if daily_bearish and rsi_bearish:
                new_signal = -SIZE_ENTRY
            elif weekly_bearish and hma_slope_negative:
                new_signal = -SIZE_ENTRY
        
        # HMA slope turn negative with KAMA confirmation
        elif hma_slope_negative and hma_slope_decreasing:
            if kama_bearish and (daily_bearish or weekly_bearish):
                new_signal = -SIZE_ENTRY
        
        # Price below both KAMA and HMA in downtrend
        elif kama_bearish and hma_slope_negative:
            if rsi_bearish and vol_normal:
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