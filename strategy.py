#!/usr/bin/env python3
"""
Experiment #049: 15m KAMA Trend with 4h/1h Multi-TF Filter + Choppiness Regime
Hypothesis: 15m failed before due to noise and fee drag. Solution: (1) Use KAMA 
instead of EMA/HMA for adaptive trend following (responds faster in trends, slower 
in ranges), (2) Require BOTH 4h AND 1h trend alignment (stronger HTF filter), 
(3) Add Choppiness Index to avoid range markets (only trade when CHOP < 50), 
(4) Use RSI pullback entries (wait for weakness in uptrend, strength in downtrend),
(5) Reduce position size to 0.25 to limit drawdown, (6) Tight 2.0*ATR stoploss.
Key insight from failures: 15m needs STRONGER filters than 12h/1d to overcome noise.
Timeframe: 15m primary, 1h and 4h for trend confirmation.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_kama_4h_1h_chop_rsi_pullback_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in ranges.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.abs(close[0:period] - close[0])
    volatility = np.abs(close - np.roll(close, 1))
    volatility[0] = change[0]
    
    vol_sum = pd.Series(volatility).rolling(window=period, min_periods=period).sum().values
    er = np.where(vol_sum > 0, change / vol_sum, 0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    tr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    chop = np.zeros(len(close))
    mask = (highest_high - lowest_low) > 0
    chop[mask] = 100 * np.log10(tr_sum[mask] / (highest_high[mask] - lowest_low[mask])) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    
    # 15m KAMA for adaptive trend
    kama_fast = calculate_kama(close, 10, 2, 30)
    kama_slow = calculate_kama(close, 30, 2, 30)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    
    for i in range(100, n):
        # HTF trend filters (MUST both align)
        hma_4h_bullish = hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]
        hma_4h_bearish = hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]
        
        hma_1h_bullish = hma_1h_aligned[i] > 0 and close[i] > hma_1h_aligned[i]
        hma_1h_bearish = hma_1h_aligned[i] > 0 and close[i] < hma_1h_aligned[i]
        
        # Both HTFs must agree
        strong_bullish = hma_4h_bullish and hma_1h_bullish
        strong_bearish = hma_4h_bearish and hma_1h_bearish
        
        # Choppiness filter (only trade in trending markets)
        trending_market = chop[i] < 50  # Below 50 = trending, above = ranging
        
        # 15m KAMA trend
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover signals
        kama_flip_long = (i > 0) and (kama_fast[i] > kama_slow[i]) and (kama_fast[i-1] <= kama_slow[i-1])
        kama_flip_short = (i > 0) and (kama_fast[i] < kama_slow[i]) and (kama_fast[i-1] >= kama_slow[i-1])
        
        # RSI pullback signals (enter on weakness in uptrend, strength in downtrend)
        rsi_pullback_long = rsi[i] < 50 and rsi[i] > 30  # Dip in uptrend
        rsi_pullback_short = rsi[i] > 50 and rsi[i] < 70  # Rally in downtrend
        rsi_rising = (i > 2) and (rsi[i] > rsi[i-2])
        rsi_falling = (i > 2) and (rsi[i] < rsi[i-2])
        
        new_signal = 0.0
        
        # LONG ENTRY: All filters must agree
        if trending_market and strong_bullish:
            # Entry on KAMA flip
            if kama_flip_long:
                new_signal = SIZE_ENTRY
            # Entry on RSI pullback with KAMA bullish
            elif kama_bullish and rsi_pullback_long and rsi_rising:
                new_signal = SIZE_ENTRY
            # Entry on KAMA bullish confirmation
            elif kama_bullish and close[i] > kama_fast[i]:
                new_signal = SIZE_ENTRY
        
        # SHORT ENTRY: All filters must agree
        if trending_market and strong_bearish:
            # Entry on KAMA flip
            if kama_flip_short:
                new_signal = -SIZE_ENTRY
            # Entry on RSI pullback with KAMA bearish
            elif kama_bearish and rsi_pullback_short and rsi_falling:
                new_signal = -SIZE_ENTRY
            # Entry on KAMA bearish confirmation
            elif kama_bearish and close[i] < kama_fast[i]:
                new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position tracking
        if position_side > 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = close[i] - entry_price
                    risk = 2.0 * atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = SIZE_HALF
                        position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Calculate trailing stop
            current_stop = close[i] + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            else:
                # Check take profit (reduce position at 2R)
                if not position_reduced:
                    profit = entry_price - close[i]
                    risk = 2.0 * atr[i]
                    if risk > 0 and profit >= 2.0 * risk:
                        new_signal = -SIZE_HALF
                        position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals