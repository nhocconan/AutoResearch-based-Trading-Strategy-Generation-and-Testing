#!/usr/bin/env python3
"""
Experiment #016: 4h Asymmetric Regime Strategy with 1d HTF Trend Filter
Hypothesis: 4h timeframe needs fewer, higher-quality signals. Using 1d KAMA
for smoother trend bias (less whipsaw than HMA). Asymmetric entry logic:
longs on RSI pullback in uptrend, shorts only when ADX strong + 1d bearish.
Bollinger Band Width for volatility regime (squeeze = breakout potential).
Conservative sizing (0.22) with 2.5*ATR stop appropriate for 4h noise.
Timeframe: 4h (REQUIRED), HTF: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_asymmetric_kama_1d_rsi_bb_adx_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise - moves fast in trends, slow in ranges.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        if i == er_period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    mid = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    bw = (upper - lower) / mid  # Bandwidth
    return upper, mid, lower, bw

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    mean = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - mean) / (std + 1e-10)
    return zscore

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish pressure)."""
    ratio = np.zeros(len(volume))
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_fast = calculate_rsi(close, 7)
    adx = calculate_adx(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower, bb_bw = calculate_bollinger(close, 20, 2.0)
    
    # KAMA for 4h trend
    kama_4h = calculate_kama(close, er_period=10)
    kama_4h_fast = calculate_kama(close, er_period=5)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.22
    SIZE_HALF = 0.11
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    # Volatility regime tracking (BB Width percentile)
    bb_bw_percentile = pd.Series(bb_bw).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x[-1]) / len(x) if len(x) >= 50 else np.nan
    ).values
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(kama_4h[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - primary filter
        htf_bullish = close[i] > kama_1d_aligned[i]
        htf_bearish = close[i] < kama_1d_aligned[i]
        
        # 1d KAMA slope
        htf_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1] if i > 0 else False
        htf_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1] if i > 0 else False
        
        # 4h KAMA trend
        kama_4h_bullish = close[i] > kama_4h[i]
        kama_4h_bearish = close[i] < kama_4h[i]
        kama_rising = kama_4h[i] > kama_4h[i-1] if i > 0 else False
        kama_falling = kama_4h[i] < kama_4h[i-1] if i > 0 else False
        
        # KAMA crossover
        fast_above_slow = kama_4h_fast[i] > kama_4h[i]
        fast_below_slow = kama_4h_fast[i] < kama_4h[i]
        
        # ADX regime
        trend_strong = adx[i] > 25
        trend_weak = adx[i] < 20
        
        # Volatility regime
        vol_squeeze = bb_bw_percentile[i] < 0.3 if not np.isnan(bb_bw_percentile[i]) else False
        vol_expansion = bb_bw_percentile[i] > 0.7 if not np.isnan(bb_bw_percentile[i]) else False
        
        # RSI conditions
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_fast_oversold = rsi_fast[i] < 25
        rsi_fast_overbought = rsi_fast[i] > 75
        rsi_neutral = 40 < rsi[i] < 60
        
        # Z-score extremes
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        # Bollinger position
        price_at_lower = close[i] < bb_lower[i] * 1.005
        price_at_upper = close[i] > bb_upper[i] * 0.995
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.52
        vol_bearish = vol_ratio[i] < 0.48
        
        new_signal = 0.0
        
        # === ASYMMETRIC LONG ENTRIES (easier to trigger) ===
        
        # Path 1: 1d bullish + 4h RSI pullback (primary long setup)
        if htf_bullish and rsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 2: 1d bullish + 4h KAMA bullish + RSI fast oversold (dip buy)
        elif htf_bullish and kama_4h_bullish and rsi_fast_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 3: 1d bullish + price at BB lower + volume bullish (bounce play)
        elif htf_bullish and price_at_lower and vol_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 4: 1d rising + 4h KAMA crossover up + ADX not weak (momentum)
        elif htf_rising and fast_above_slow and not trend_weak:
            new_signal = SIZE_ENTRY
        
        # Path 5: Z-score oversold + 1d bullish (mean reversion in uptrend)
        elif zscore_oversold and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 6: Volatility squeeze breakout long
        elif vol_squeeze and close[i] > bb_mid[i] and vol_bullish:
            new_signal = SIZE_ENTRY
        
        # === ASYMMETRIC SHORT ENTRIES (harder to trigger - bear market only) ===
        
        # Path 1: 1d bearish + 4h RSI overbought (primary short setup)
        if htf_bearish and rsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 1d bearish + 4h KAMA bearish + RSI fast overbought (rally sell)
        elif htf_bearish and kama_4h_bearish and rsi_fast_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 1d bearish + price at BB upper + volume bearish (rejection)
        elif htf_bearish and price_at_upper and vol_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 1d falling + 4h KAMA crossover down + ADX strong (momentum short)
        elif htf_falling and fast_below_slow and trend_strong:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Z-score overbought + 1d bearish (mean reversion in downtrend)
        elif zscore_overbought and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
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