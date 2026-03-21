#!/usr/bin/env python3
"""
EXPERIMENT #003 - KAMA Adaptive Trend + MACD Momentum Strategy (1h)
====================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility
better than HMA/EMA, reducing whipsaws in choppy markets. Combined with MACD
histogram for momentum confirmation and ADX for trend strength filtering, this
should capture trending moves while avoiding range-bound losses.

Key differences from failed #001/#002:
- KAMA instead of HMA (adaptive to volatility, fewer false signals)
- MACD histogram for momentum confirmation (not just RSI pullback)
- ADX filter: only trade when ADX > 25 (strong trend)
- 1h primary timeframe (this experiment's rotation)
- 4h KAMA for trend direction filter
- Smaller position size (0.25 max) for better drawdown control
- Proper stoploss at 2*ATR with signal→0 exit

Primary TF: 1h | HTF: 4h KAMA trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_macd_adx_1h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close: np.ndarray, er_period: int = 10, fast_sc: int = 2, slow_sc: int = 30) -> np.ndarray:
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market efficiency ratio (ER)
    High ER (trending) = fast smoothing, Low ER (choppy) = slow smoothing
    """
    n = len(close)
    if n < er_period + slow_sc:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = close_s.diff(er_period).abs()
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Calculate Smoothing Constant (SC)
    fast_sc_val = 2 / (fast_sc + 1)
    slow_sc_val = 2 / (slow_sc + 1)
    sc = (er * (fast_sc_val - slow_sc_val) + slow_sc_val) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize with price
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    kama[:er_period] = np.nan
    return kama


def calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Calculate MACD line, signal line, and histogram
    Returns: (macd_line, signal_line, histogram)
    """
    n = len(close)
    if n < slow + signal:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    close_s = pd.Series(close)
    
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    
    macd_line = (ema_fast - ema_slow).values
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    histogram[:slow + signal] = np.nan
    return macd_line, signal_line, histogram


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index (ADX)
    ADX > 25 indicates strong trend, ADX < 20 indicates range
    """
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    atr = tr.rolling(window=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=period).mean() / atr)
    
    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window=period, min_periods=period).mean()
    
    adx_values = adx.values
    adx_values[:period * 2] = np.nan
    return adx_values


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    # Extract price data
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (CRITICAL RULE #1) ===
    df_4h = get_htf_data(prices, '4h')
    kama_4h = calculate_kama(df_4h['close'].values, er_period=10, fast_sc=2, slow_sc=30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)  # auto shift(1)
    
    # === CALCULATE 1h INDICATORS (vectorized before loop) ===
    kama_1h = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # === SIGNAL PARAMETERS ===
    SIZE_ENTRY = 0.25      # 25% position on entry (conservative)
    SIZE_HALF = 0.125      # 12.5% after take profit
    STOPLOSS_MULT = 2.0    # 2*ATR stoploss
    TAKEPROFIT_MULT = 2.0  # 2R take profit
    ADX_THRESHOLD = 25     # Only trade when ADX > 25 (strong trend)
    
    signals = np.zeros(n)
    
    # Track position state for stoploss/takeprofit
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    trailing_stop = 0.0
    
    min_lookback = max(100, 30)  # Ensure all indicators are valid
    
    for i in range(min_lookback, n):
        # Skip if any indicator is NaN
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(kama_1h[i]) or 
            np.isnan(macd_hist[i]) or np.isnan(adx[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
            continue
        
        current_atr = atr[i]
        current_price = close[i]
        current_adx = adx[i]
        current_kama_4h = kama_4h_aligned[i]
        current_kama_1h = kama_1h[i]
        current_macd_hist = macd_hist[i]
        
        # === TREND FILTER (4h KAMA) ===
        # Price above 4h KAMA = uptrend, below = downtrend
        trend_up = current_price > current_kama_4h
        trend_down = current_price < current_kama_4h
        
        # === TREND STRENGTH FILTER (ADX) ===
        strong_trend = current_adx > ADX_THRESHOLD
        
        # === MOMENTUM FILTER (MACD Histogram) ===
        # MACD hist > 0 and rising = bullish momentum
        # MACD hist < 0 and falling = bearish momentum
        macd_bullish = current_macd_hist > 0
        macd_bearish = current_macd_hist < 0
        
        # Check MACD histogram direction (compare to previous)
        if i > min_lookback:
            prev_macd_hist = macd_hist[i-1]
            macd_rising = current_macd_hist > prev_macd_hist
            macd_falling = current_macd_hist < prev_macd_hist
        else:
            macd_rising = False
            macd_falling = False
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        if position_side == 0:
            # === LONG ENTRY: uptrend + strong ADX + MACD bullish ===
            if trend_up and strong_trend and macd_bullish and macd_rising:
                # Confirm 1h KAMA also supportive (price above KAMA)
                if current_price > current_kama_1h:
                    new_signal = SIZE_ENTRY
                    entry_price = current_price
                    position_side = 1
                    highest_since_entry = current_price
                    lowest_since_entry = current_price
                    trailing_stop = entry_price - STOPLOSS_MULT * current_atr
            
            # === SHORT ENTRY: downtrend + strong ADX + MACD bearish ===
            elif trend_down and strong_trend and macd_bearish and macd_falling:
                # Confirm 1h KAMA also supportive (price below KAMA)
                if current_price < current_kama_1h:
                    new_signal = -SIZE_ENTRY
                    entry_price = current_price
                    position_side = -1
                    highest_since_entry = current_price
                    lowest_since_entry = current_price
                    trailing_stop = entry_price + STOPLOSS_MULT * current_atr
        
        elif position_side == 1:
            # Track highest price since entry for trailing
            highest_since_entry = max(highest_since_entry, current_price)
            
            # === TRAILING STOP: move stop up as price rises ===
            new_trailing_stop = highest_since_entry - STOPLOSS_MULT * current_atr
            if new_trailing_stop > trailing_stop:
                trailing_stop = new_trailing_stop
            
            # === STOPLOSS: price drops below trailing stop ===
            if current_price < trailing_stop:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif entry_price > 0:
                profit_r = (current_price - entry_price) / current_atr
                if profit_r >= TAKEPROFIT_MULT:
                    new_signal = SIZE_HALF
                    # Update trailing stop to lock in profit
                    trailing_stop = entry_price + 0.5 * current_atr
            
            # === EXIT: MACD momentum reversal ===
            elif macd_bearish and macd_falling:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: Trend reversal (4h KAMA) ===
            elif trend_down:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            else:
                new_signal = SIZE_ENTRY if new_signal == 0 else new_signal
        
        elif position_side == -1:
            # Track lowest price since entry for trailing
            lowest_since_entry = min(lowest_since_entry, current_price)
            
            # === TRAILING STOP: move stop down as price falls ===
            new_trailing_stop = lowest_since_entry + STOPLOSS_MULT * current_atr
            if new_trailing_stop < trailing_stop or trailing_stop == 0:
                trailing_stop = new_trailing_stop
            
            # === STOPLOSS: price rises above trailing stop ===
            if current_price > trailing_stop:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === TAKE PROFIT: at 2R, reduce to half ===
            elif entry_price > 0:
                profit_r = (entry_price - current_price) / current_atr
                if profit_r >= TAKEPROFIT_MULT:
                    new_signal = -SIZE_HALF
                    # Update trailing stop to lock in profit
                    trailing_stop = entry_price - 0.5 * current_atr
            
            # === EXIT: MACD momentum reversal ===
            elif macd_bullish and macd_rising:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            # === EXIT: Trend reversal (4h KAMA) ===
            elif trend_up:
                new_signal = 0.0
                position_side = 0
                entry_price = 0.0
                trailing_stop = 0.0
            
            else:
                new_signal = -SIZE_ENTRY if new_signal == 0 else new_signal
        
        signals[i] = new_signal
    
    return signals