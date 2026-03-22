#!/usr/bin/env python3
"""
Experiment #481: 15m KAMA Trend + 4h HMA Bias + 1h ADX Regime + RSI Pullback + ATR Stop
Hypothesis: 15m timeframe needs strong HTF filters to avoid noise. Using 4h HMA for 
trend bias + 1h ADX for regime detection + 15m KAMA (adaptive) for entries. KAMA 
adapts to volatility better than EMA/HMA, reducing whipsaws. Multiple entry paths 
ensure >=10 trades per symbol. RSI pullback (35-65 range) catches entries without 
being too strict. 2*ATR stoploss appropriate for 15m bars.
Timeframe: 15m (REQUIRED), HTF: 1h and 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_kama_4h_hma_1h_adx_rsi_pullback_atr_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average - adapts to market noise."""
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[0:period] = np.nan
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1)))
    
    er = np.zeros(n)
    er[period:] = np.where(volatility[period:] > 0, change[period:] / volatility[period:], 0)
    
    # Smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
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
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator for trend direction."""
    atr = calculate_atr(high, low, close, period)
    n = len(close)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            trend[i] = 1
        else:
            supertrend[i] = upper_band[i]
            trend[i] = -1
    
    return supertrend, trend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(
        df_1h['high'].values, 
        df_1h['low'].values, 
        df_1h['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    plus_di_1h_aligned = align_htf_to_ltf(prices, df_1h, plus_di_1h)
    minus_di_1h_aligned = align_htf_to_ltf(prices, df_1h, minus_di_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    kama_15m = calculate_kama(close, period=10)
    kama_15m_fast = calculate_kama(close, period=5)
    rsi = calculate_rsi(close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
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
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(kama_15m[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_trend[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h ADX regime (trending vs ranging)
        adx_trending = adx_1h_aligned[i] > 15  # Lower threshold for more trades
        adx_strong = adx_1h_aligned[i] > 25
        
        # 1h DI direction
        di_1h_bullish = plus_di_1h_aligned[i] > minus_di_1h_aligned[i]
        di_1h_bearish = plus_di_1h_aligned[i] < minus_di_1h_aligned[i]
        
        # 15m KAMA trend
        kama_bullish = close[i] > kama_15m[i]
        kama_bearish = close[i] < kama_15m[i]
        kama_rising = kama_15m[i] > kama_15m[i-1] if i > 0 else False
        kama_falling = kama_15m[i] < kama_15m[i-1] if i > 0 else False
        
        # Fast KAMA crossover
        fast_above_slow = kama_15m_fast[i] > kama_15m[i]
        fast_below_slow = kama_15m_fast[i] < kama_15m[i]
        
        # Supertrend direction
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # RSI zones (wider bands for more trades)
        rsi_bullish = rsi[i] > 35 and rsi[i] < 65
        rsi_bearish = rsi[i] > 35 and rsi[i] < 65
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bullish + 1h trending + 15m KAMA bullish + RSI ok
        if trend_4h_bullish and adx_trending and kama_bullish and rsi_bullish:
            new_signal = SIZE_ENTRY
        # Path 2: 4h bullish + 1h DI bullish + Supertrend bullish
        elif trend_4h_bullish and di_1h_bullish and st_bullish:
            new_signal = SIZE_ENTRY
        # Path 3: 15m KAMA bullish + KAMA rising + Fast above slow + RSI oversold
        elif kama_bullish and kama_rising and fast_above_slow and rsi_oversold:
            new_signal = SIZE_ENTRY
        # Path 4: Supertrend bullish + 4h bullish + ADX > 15
        elif st_bullish and trend_4h_bullish and adx_1h_aligned[i] > 15:
            new_signal = SIZE_ENTRY
        # Path 5: Price above KAMA + Fast KAMA crossover up + RSI 40-60
        elif close[i] > kama_15m[i] and fast_above_slow and rsi[i] > 40 and rsi[i] < 60:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: 4h bearish + 1h trending + 15m KAMA bearish + RSI ok
        if trend_4h_bearish and adx_trending and kama_bearish and rsi_bearish:
            new_signal = -SIZE_ENTRY
        # Path 2: 4h bearish + 1h DI bearish + Supertrend bearish
        elif trend_4h_bearish and di_1h_bearish and st_bearish:
            new_signal = -SIZE_ENTRY
        # Path 3: 15m KAMA bearish + KAMA falling + Fast below slow + RSI overbought
        elif kama_bearish and kama_falling and fast_below_slow and rsi_overbought:
            new_signal = -SIZE_ENTRY
        # Path 4: Supertrend bearish + 4h bearish + ADX > 15
        elif st_bearish and trend_4h_bearish and adx_1h_aligned[i] > 15:
            new_signal = -SIZE_ENTRY
        # Path 5: Price below KAMA + Fast KAMA crossover down + RSI 40-60
        elif close[i] < kama_15m[i] and fast_below_slow and rsi[i] > 40 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR for 15m timeframe)
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
            
            # Calculate trailing stop (2*ATR for 15m timeframe)
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
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
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