#!/usr/bin/env python3
"""
Experiment #011: 12h Adaptive Trend with 1d HMA Regime Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts better to volatility changes than EMA.
Combined with 1d HMA for regime bias and ADX for trend strength, this should capture
sustained moves while avoiding choppy periods. 12h timeframe reduces whipsaw vs 1h/4h.
Key changes from exp#005: KAMA instead of EMA, ADX filter (>20 not >25), simpler RSI conditions.
Position sizing: 0.30 discrete levels, 2.5*ATR trailing stop.
Must generate 10+ trades on train, 3+ on test - conditions loosened vs previous attempts.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_1d_hma_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average - adapts to market volatility.
    ER (Efficiency Ratio) determines smoothing constant.
    High ER (trending) = fast smoothing, Low ER (choppy) = slow smoothing.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    er = change / (volatility + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Dynamic smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    if n < period * 2:
        return adx
    
    # Calculate +DM and -DM
    high_diff = high - np.roll(high, 1)
    low_diff = np.roll(low, 1) - low
    high_diff[0] = 0
    low_diff[0] = 0
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Smooth with Wilder's method
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_di / (atr + 1e-10)
    minus_di = 100 * minus_di / (atr + 1e-10)
    
    # Calculate DX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = 100 * di_diff / (di_sum + 1e-10)
    
    # Calculate ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period):
    """Calculate SMA."""
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
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # KAMA for adaptive trend
    kama_fast = calculate_kama(close, 10, 2, 30)
    kama_slow = calculate_kama(close, 20, 2, 30)
    
    # SMA for long-term filter
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_LONG = 0.30
    SIZE_SHORT = -0.30
    SIZE_HALF_LONG = 0.15
    SIZE_HALF_SHORT = -0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or adx[i] == 0:
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 12h KAMA trend
        kama_bull = kama_fast[i] > kama_slow[i]
        kama_bear = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover signals
        kama_cross_long = False
        kama_cross_short = False
        if i >= 1 and not np.isnan(kama_fast[i]) and not np.isnan(kama_fast[i-1]):
            kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
            kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # ADX trend strength (>20 = trending, <20 = choppy)
        trending = adx[i] > 20
        
        # DI crossover for direction
        di_bull = plus_di[i] > minus_di[i]
        di_bear = plus_di[i] < minus_di[i]
        
        # RSI conditions - LOOSENED for more trades
        rsi_bullish = rsi[i] > 40 and rsi[i] < 70
        rsi_bearish = rsi[i] > 30 and rsi[i] < 60
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # Long-term filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Price position relative to KAMA
        price_above_kama = close[i] > kama_fast[i]
        price_below_kama = close[i] < kama_fast[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 1d bullish) ===
        if bull_trend_1d:
            # Primary: KAMA crossover with ADX confirmation
            if kama_cross_long and trending and di_bull:
                new_signal = SIZE_LONG
            
            # Secondary: Pullback to KAMA in uptrend
            elif price_above_kama and kama_bull and rsi_bullish and above_200:
                new_signal = SIZE_LONG
            
            # Tertiary: RSI oversold bounce with trend
            elif rsi_oversold and kama_bull and bull_trend_1d:
                new_signal = SIZE_HALF_LONG
            
            # Momentum: DI bullish crossover
            elif di_bull and kama_bull and rsi[i] > 45:
                new_signal = SIZE_HALF_LONG
        
        # === SHORT ENTRIES (only when 1d bearish) ===
        elif bear_trend_1d:
            # Primary: KAMA crossover with ADX confirmation
            if kama_cross_short and trending and di_bear:
                new_signal = SIZE_SHORT
            
            # Secondary: Bounce to KAMA in downtrend
            elif price_below_kama and kama_bear and rsi_bearish and below_200:
                new_signal = SIZE_SHORT
            
            # Tertiary: RSI overbought rejection with trend
            elif rsi_overbought and kama_bear and bear_trend_1d:
                new_signal = SIZE_HALF_SHORT
            
            # Momentum: DI bearish crossover
            elif di_bear and kama_bear and rsi[i] < 55:
                new_signal = SIZE_HALF_SHORT
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            # New entry
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            # Reversal
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            # Exit
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals