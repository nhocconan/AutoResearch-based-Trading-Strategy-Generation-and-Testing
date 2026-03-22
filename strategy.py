#!/usr/bin/env python3
"""
Experiment #470: 30m KAMA Adaptive Trend + 4h HMA Bias + RSI Pullback + ATR Stop
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast in trends,
slow in ranges. This should reduce whipsaw on 30m compared to fixed EMA/HMA.
4h HMA provides trend bias filter. RSI pullback (40-60 zone) ensures we enter on dips,
not chasing. 2.5*ATR stoploss protects capital. Multiple entry paths ensure >=10 trades.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_hma_rsi_pullback_atr_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise - moves fast in trends, slow in ranges.
    Formula from Perry Kaufman's "Trading Systems and Methods".
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio (ER)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er = signal / noise
        else:
            er = 0.0
        
        # Smoothing constant
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        if i == period:
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_slope(values, lookback=5):
    """Calculate slope of values over lookback period."""
    n = len(values)
    slope = np.zeros(n)
    slope[:] = np.nan
    for i in range(lookback, n):
        if not np.isnan(values[i]) and not np.isnan(values[i - lookback]):
            slope[i] = (values[i] - values[i - lookback]) / lookback
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, period=10)
    kama_fast = calculate_kama(close, period=6, fast_period=2, slow_period=20)
    rsi = calculate_rsi(close, 14)
    kama_slope = calculate_slope(kama, lookback=5)
    
    # Simple moving average for additional filter
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(kama_slope[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        h4h_bullish = close[i] > hma_4h_aligned[i]
        h4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 30m KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        kama_rising = kama_slope[i] > 0
        kama_falling = kama_slope[i] < 0
        
        # Fast KAMA crossover
        fast_above_slow = kama_fast[i] > kama[i]
        fast_below_slow = kama_fast[i] < kama[i]
        
        # Price vs SMA50
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # RSI pullback zones (wider for more trades)
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 65
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 60
        rsi_neutral_short = rsi[i] > 40 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bullish + 30m KAMA bullish + RSI pullback + KAMA rising
        if h4h_bullish and kama_bullish and rsi_pullback_long and kama_rising:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + Fast KAMA above slow + RSI neutral + above SMA50
        elif h4h_bullish and fast_above_slow and rsi_neutral_long and above_sma50:
            new_signal = SIZE_ENTRY
        # Path 3: 30m KAMA bullish + KAMA rising + RSI > 40 (simpler entry)
        elif kama_bullish and kama_rising and rsi[i] > 40 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        # Path 4: 4h bullish + 30m KAMA bullish + Fast KAMA crossover up
        elif h4h_bullish and kama_bullish and fast_above_slow and kama_fast[i] > kama_fast[i-1]:
            new_signal = SIZE_ENTRY
        # Path 5: Price above KAMA + above SMA50 + RSI 45-55 (consolidation breakout)
        elif close[i] > kama[i] and above_sma50 and rsi[i] > 45 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bearish + 30m KAMA bearish + RSI pullback + KAMA falling
        if h4h_bearish and kama_bearish and rsi_pullback_short and kama_falling:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + Fast KAMA below slow + RSI neutral + below SMA50
        elif h4h_bearish and fast_below_slow and rsi_neutral_short and below_sma50:
            new_signal = -SIZE_ENTRY
        # Path 3: 30m KAMA bearish + KAMA falling + RSI < 60 (simpler entry)
        elif kama_bearish and kama_falling and rsi[i] > 45 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 4: 4h bearish + 30m KAMA bearish + Fast KAMA crossover down
        elif h4h_bearish and kama_bearish and fast_below_slow and kama_fast[i] < kama_fast[i-1]:
            new_signal = -SIZE_ENTRY
        # Path 5: Price below KAMA + below SMA50 + RSI 45-55 (consolidation breakdown)
        elif close[i] < kama[i] and below_sma50 and rsi[i] > 45 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 30m timeframe)
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