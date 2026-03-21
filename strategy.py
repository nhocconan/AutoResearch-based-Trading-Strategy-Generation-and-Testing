#!/usr/bin/env python3
"""
Experiment #169: 15m KAMA-RSI Strategy with 4h/1h HMA Trend Filter
Hypothesis: 15m timeframe captures intraday swings while 4h/1h HMA provides 
trend bias to avoid counter-trend trades. KAMA (Kaufman Adaptive MA) adapts 
to volatility - fast in trends, slow in ranges. RSI pullback entries in trend 
direction reduce whipsaws. This targets the noise of 15m while respecting 
higher timeframe structure. Position sizing: 0.25 entry, 0.125 at 2R profit.
ATR stoploss at 2.5*ATR to survive 15m volatility.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_kama_rsi_4h_1h_hma_trend_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise - fast during trends, slow during ranges.
    Reference: Perry Kaufman, "Trading Systems and Methods"
    """
    close_s = pd.Series(close)
    
    # Change = absolute price change over er_period
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.abs(close[:er_period] - close[0])
    
    # Volatility = sum of absolute price changes over er_period
    volatility = np.zeros(len(close))
    for i in range(er_period, len(close)):
        volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1)))
    volatility[:er_period] = change[:er_period]
    
    # Efficiency Ratio (ER)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility != 0)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = (2.0 / (fast_period + 1)) ** 2
    slow_sc = (2.0 / (slow_period + 1)) ** 2
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    return upper, lower, mid

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
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=10)
    kama_slow = calculate_kama(close, er_period=20, fast_period=5, slow_period=30)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, 20)
    
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
        # HTF trend filters (4h primary, 1h confirmation)
        hma_4h_valid = hma_4h_aligned[i] > 0 and not np.isnan(hma_4h_aligned[i])
        hma_1h_valid = hma_1h_aligned[i] > 0 and not np.isnan(hma_1h_aligned[i])
        
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        trend_1h_bullish = hma_1h_valid and close[i] > hma_1h_aligned[i]
        trend_1h_bearish = hma_1h_valid and close[i] < hma_1h_aligned[i]
        
        # 15m KAMA trend
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover signals
        kama_cross_long = kama_bullish and kama_fast[i-1] <= kama_slow[i-1] if i > 0 else False
        kama_cross_short = kama_bearish and kama_fast[i-1] >= kama_slow[i-1] if i > 0 else False
        
        # RSI signals (pullback in trend direction)
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else False
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else False
        
        # Donchian breakout
        donch_breakout_long = close[i] > donch_upper[i-1] if i > 0 else False
        donch_breakout_short = close[i] < donch_lower[i-1] if i > 0 else False
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 4h trend bullish + 15m KAMA crossover + RSI not overbought
        if trend_4h_bullish and kama_cross_long:
            if rsi[i] < 65:  # Not extremely overbought
                new_signal = SIZE_ENTRY
        
        # Secondary: 4h bullish + 1h bullish + RSI pullback + KAMA bullish
        elif trend_4h_bullish and trend_1h_bullish:
            if rsi_oversold and rsi_rising and kama_bullish:
                new_signal = SIZE_ENTRY
        
        # Tertiary: Donchian breakout with HTF confirmation
        elif trend_4h_bullish and donch_breakout_long:
            if kama_bullish:
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 4h trend bearish + 15m KAMA crossover + RSI not oversold
        if trend_4h_bearish and kama_cross_short:
            if rsi[i] > 35:  # Not extremely oversold
                new_signal = -SIZE_ENTRY
        
        # Secondary: 4h bearish + 1h bearish + RSI pullback + KAMA bearish
        elif trend_4h_bearish and trend_1h_bearish:
            if rsi_overbought and rsi_falling and kama_bearish:
                new_signal = -SIZE_ENTRY
        
        # Tertiary: Donchian breakdown with HTF confirmation
        elif trend_4h_bearish and donch_breakout_short:
            if kama_bearish:
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