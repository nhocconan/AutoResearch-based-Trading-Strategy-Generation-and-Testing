#!/usr/bin/env python3
"""
Experiment #410: 30m HMA Trend + 4h Regime + RSI Pullback + ATR Stop
Hypothesis: Previous 30m Supertrend strategies (#398, #404) failed due to whipsaw in choppy markets.
Switch to HMA crossover (smoother than EMA) with 4h HMA as regime filter (not strict entry requirement).
RSI pullback entries within trend direction reduce counter-trend losses. ADX filter avoids low-volatility chop.
Multiple entry paths ensure >=10 trades per symbol (learned from 0-trade failures in #400, #402, #407, #409).
Key difference: HMA crossover instead of Supertrend, ADX>20 filter to avoid chop, softer 4h bias.
Position size: 0.28 discrete, stoploss 2*ATR for 30m timeframe.
Timeframe: 30m (REQUIRED), HTF: 4h for regime bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_trend_4h_regime_rsi_pullback_adx_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_diff = high[i] - high[i-1]
        minus_diff = low[i-1] - low[i]
        
        if plus_diff > minus_diff and plus_diff > 0:
            plus_dm[i] = plus_diff
        elif minus_diff > plus_diff and minus_diff > 0:
            minus_dm[i] = minus_diff
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_fast = calculate_hma(df_4h['close'].values, 16)
    hma_4h_slow = calculate_hma(df_4h['close'].values, 48)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_fast)
    hma_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slow)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # HMA crossover on 30m
    hma_30m_fast = calculate_hma(close, 16)
    hma_30m_slow = calculate_hma(close, 48)
    
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
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_fast_aligned[i]) or np.isnan(hma_30m_fast[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend regime (SOFT bias, not strict filter)
        hma_4h_bullish = hma_4h_fast_aligned[i] > hma_4h_slow_aligned[i]
        hma_4h_bearish = hma_4h_fast_aligned[i] < hma_4h_slow_aligned[i]
        
        # 30m HMA crossover signals
        hma_30m_bull_cross = hma_30m_fast[i] > hma_30m_slow[i] and hma_30m_fast[i-1] <= hma_30m_slow[i-1]
        hma_30m_bear_cross = hma_30m_fast[i] < hma_30m_slow[i] and hma_30m_fast[i-1] >= hma_30m_slow[i-1]
        
        # HMA trend direction (already crossed)
        hma_30m_uptrend = hma_30m_fast[i] > hma_30m_slow[i]
        hma_30m_downtrend = hma_30m_fast[i] < hma_30m_slow[i]
        
        # ADX trend strength filter (avoid chop when ADX < 20)
        adx_strong = adx[i] > 20
        
        # RSI pullback levels (enter on pullback within trend)
        rsi_pullback_long = rsi[i] >= 35 and rsi[i] <= 55  # Pullback in uptrend
        rsi_pullback_short = rsi[i] >= 45 and rsi[i] <= 65  # Pullback in downtrend
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 75
        rsi_momentum_short = rsi[i] > 25 and rsi[i] < 55
        
        # DI crossover confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: HMA bull cross + ADX strong + RSI momentum + 4h bullish bias
        if hma_30m_bull_cross and adx_strong and rsi_momentum_long and hma_4h_bullish:
            new_signal = SIZE_ENTRY
        # Path 2: HMA uptrend + RSI pullback + DI bullish + ADX ok
        elif hma_30m_uptrend and rsi_pullback_long and di_bullish and adx[i] > 15:
            new_signal = SIZE_ENTRY
        # Path 3: HMA bull cross + DI bullish + 4h bullish (ADX neutral ok)
        elif hma_30m_bull_cross and di_bullish and hma_4h_bullish:
            new_signal = SIZE_ENTRY
        # Path 4: HMA uptrend + RSI > 50 + DI bullish + 4h bullish
        elif hma_30m_uptrend and rsi[i] > 50 and di_bullish and hma_4h_bullish:
            new_signal = SIZE_ENTRY
        # Path 5: HMA bull cross + RSI > 45 (simpler entry for more trades)
        elif hma_30m_bull_cross and rsi[i] > 45 and adx[i] > 18:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths to ensure >=10 trades) ===
        # Path 1: HMA bear cross + ADX strong + RSI momentum + 4h bearish bias
        if hma_30m_bear_cross and adx_strong and rsi_momentum_short and hma_4h_bearish:
            new_signal = -SIZE_ENTRY
        # Path 2: HMA downtrend + RSI pullback + DI bearish + ADX ok
        elif hma_30m_downtrend and rsi_pullback_short and di_bearish and adx[i] > 15:
            new_signal = -SIZE_ENTRY
        # Path 3: HMA bear cross + DI bearish + 4h bearish (ADX neutral ok)
        elif hma_30m_bear_cross and di_bearish and hma_4h_bearish:
            new_signal = -SIZE_ENTRY
        # Path 4: HMA downtrend + RSI < 50 + DI bearish + 4h bearish
        elif hma_30m_downtrend and rsi[i] < 50 and di_bearish and hma_4h_bearish:
            new_signal = -SIZE_ENTRY
        # Path 5: HMA bear cross + RSI < 55 (simpler entry for more trades)
        elif hma_30m_bear_cross and rsi[i] < 55 and adx[i] > 18:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR)
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
            
            # Calculate trailing stop (2*ATR)
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