#!/usr/bin/env python3
"""
Experiment #011: 12h KAMA Adaptive Trend + 1d HMA Filter + RSI/Bollinger Entries
Hypothesis: 12h timeframe captures multi-week swings with less noise than lower TFs.
KAMA (Kaufman Adaptive MA) adapts to volatility - fast in trends, slow in ranges.
1d HMA provides smoother HTF trend bias. RSI extremes + BB bands for mean reversion
entries within trend direction. ADX filter avoids choppy markets. 3*ATR stoploss
for 12h bars (wider than lower TFs). Multiple entry paths (6 long + 6 short) to
ensure >=10 trades per symbol despite slower 12h frequency. Timeframe: 12h (REQUIRED), HTF: 1d.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_1d_hma_rsi_bb_adx_atr_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < slow_period:
        return kama
    
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.zeros(n)
    er[:] = np.nan
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    er[:er_period] = np.nan
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = np.zeros(n)
    sc[:] = np.nan
    valid_er = ~np.isnan(er)
    sc[valid_er] = (er[valid_er] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    mid = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    bw = (upper - lower) / (mid + 1e-10)  # Bandwidth
    return upper, lower, mid, bw

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

def calculate_momentum(close, period=10):
    """Calculate Rate of Change (ROC) momentum."""
    momentum = np.zeros(len(close))
    momentum[:] = np.nan
    for i in range(period, len(close)):
        if close[i-period] > 0:
            momentum[i] = (close[i] - close[i-period]) / close[i-period] * 100
    return momentum

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
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    bb_upper, bb_lower, bb_mid, bb_bw = calculate_bollinger(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    momentum = calculate_momentum(close, 10)
    
    # Fast KAMA for crossover signals
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - LOOSE filter to ensure trades
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # 12h KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        kama_rising = kama[i] > kama[i-1] if i > 0 and not np.isnan(kama[i-1]) else False
        kama_falling = kama[i] < kama[i-1] if i > 0 and not np.isnan(kama[i-1]) else False
        
        # Fast KAMA crossover
        fast_above_slow = kama_fast[i] > kama[i] if not np.isnan(kama_fast[i]) else False
        fast_below_slow = kama_fast[i] < kama[i] if not np.isnan(kama_fast[i]) else False
        
        # ADX regime - LOOSE threshold for 12h
        trend_strong = adx[i] > 18
        trend_weak = adx[i] < 28
        
        # RSI extremes - LOOSE threshold
        rsi_oversold = rsi[i] < 38
        rsi_overbought = rsi[i] > 62
        
        # Bollinger Band position
        price_near_lower = close[i] < bb_lower[i] * 1.02  # Within 2% of lower band
        price_near_upper = close[i] > bb_upper[i] * 0.98  # Within 2% of upper band
        price_below_mid = close[i] < bb_mid[i]
        price_above_mid = close[i] > bb_mid[i]
        
        # Momentum confirmation
        mom_positive = momentum[i] > 0 if not np.isnan(momentum[i]) else False
        mom_negative = momentum[i] < 0 if not np.isnan(momentum[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (6 paths for >=10 trades) ===
        
        # Path 1: 1d bullish + 12h KAMA bullish + Fast KAMA crossover up (trend entry)
        if htf_bullish and kama_bullish and fast_above_slow and kama_rising:
            new_signal = SIZE_ENTRY
        
        # Path 2: 1d bullish + RSI oversold + price near BB lower (dip buy in uptrend)
        elif htf_bullish and rsi_oversold and price_near_lower:
            new_signal = SIZE_ENTRY
        
        # Path 3: 1d bullish + ADX strong + momentum positive (trend continuation)
        elif htf_bullish and trend_strong and mom_positive:
            new_signal = SIZE_ENTRY
        
        # Path 4: 1d bullish + KAMA rising + price above BB mid (breakout confirmation)
        elif htf_bullish and kama_rising and price_above_mid:
            new_signal = SIZE_ENTRY
        
        # Path 5: RSI oversold + price near BB lower + ADX weak (mean reversion in range)
        elif rsi_oversold and price_near_lower and trend_weak:
            new_signal = SIZE_ENTRY
        
        # Path 6: Fast KAMA crossover + momentum positive + 1d bullish
        elif fast_above_slow and mom_positive and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (6 paths for >=10 trades) ===
        
        # Path 1: 1d bearish + 12h KAMA bearish + Fast KAMA crossover down (trend entry)
        if htf_bearish and kama_bearish and fast_below_slow and kama_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 2: 1d bearish + RSI overbought + price near BB upper (rally sell in downtrend)
        elif htf_bearish and rsi_overbought and price_near_upper:
            new_signal = -SIZE_ENTRY
        
        # Path 3: 1d bearish + ADX strong + momentum negative (trend continuation)
        elif htf_bearish and trend_strong and mom_negative:
            new_signal = -SIZE_ENTRY
        
        # Path 4: 1d bearish + KAMA falling + price below BB mid (breakdown confirmation)
        elif htf_bearish and kama_falling and price_below_mid:
            new_signal = -SIZE_ENTRY
        
        # Path 5: RSI overbought + price near BB upper + ADX weak (mean reversion in range)
        elif rsi_overbought and price_near_upper and trend_weak:
            new_signal = -SIZE_ENTRY
        
        # Path 6: Fast KAMA crossover down + momentum negative + 1d bearish
        elif fast_below_slow and mom_negative and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR for 12h timeframe - wider stops)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3*ATR for 12h timeframe)
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
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
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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