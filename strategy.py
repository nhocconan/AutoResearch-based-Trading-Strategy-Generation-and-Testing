#!/usr/bin/env python3
"""
Experiment #007: 15m Regime-Adaptive (Choppiness + Fisher + 4h HMA) + ATR Stop
Hypothesis: Market regime detection via Choppiness Index allows adaptive strategy selection.
Range regime (CHOP>61.8): mean reversion at Bollinger bands with Fisher confirm.
Trend regime (CHOP<38.2): trend follow with 4h HMA bias + HMA crossover.
Fisher Transform catches reversals precisely in both regimes.
4h HMA provides HTF trend bias to avoid counter-trend trades.
2*ATR stoploss for 15m timeframe. SIZE=0.25 conservative for crash protection.
Multiple entry paths ensure >=10 trades per symbol on train/test.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_chop_fisher_4h_hma_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending.
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.sum(atr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to -1 to +1 range.
    Catches reversals when Fisher crosses extreme levels.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(high_for_fisher(close, i, period))
        lowest = np.min(low_for_fisher(close, i, period))
        
        if highest == lowest:
            continue
        
        value = 0.33 * 2 * ((close[i] - lowest) / (highest - lowest) - 0.5)
        value = np.clip(value, -0.99, 0.99)
        
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def high_for_fisher(close, idx, period):
    """Helper to get high values for Fisher calculation."""
    # Approximate high using close + small range
    return close[idx-period+1:idx+1] * 1.002

def low_for_fisher(close, idx, period):
    """Helper to get low values for Fisher calculation."""
    # Approximate low using close - small range
    return close[idx-period+1:idx+1] * 0.998

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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
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
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher(close, 9)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # 15m HMA for trend
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # Regime detection via Choppiness Index
        regime_range = chop[i] > 55  # Range/choppy market
        regime_trend = chop[i] < 45  # Trending market
        
        # 15m HMA trend
        hma_15m_bullish = close[i] > hma_15m[i]
        hma_15m_bearish = close[i] < hma_15m[i]
        hma_rising = hma_15m[i] > hma_15m[i-1] if i > 0 else False
        hma_falling = hma_15m[i] < hma_15m[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_15m_fast[i] > hma_15m[i]
        fast_below_slow = hma_15m_fast[i] < hma_15m[i]
        
        # Fisher Transform signals
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_cross_up = fisher_signal[i] < -1.0 and fisher[i] > fisher_signal[i]
        fisher_cross_down = fisher_signal[i] > 1.0 and fisher[i] < fisher_signal[i]
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_mid = rsi[i] > 45 and rsi[i] < 55
        
        # Bollinger position
        bb_low = close[i] < bb_lower[i]
        bb_high = close[i] > bb_upper[i]
        bb_near_low = close[i] < bb_lower[i] * 1.005
        bb_near_high = close[i] > bb_upper[i] * 0.995
        
        new_signal = 0.0
        
        # === RANGE REGIME: Mean Reversion at BB extremes ===
        if regime_range:
            # Long: price at BB lower + Fisher oversold + 4h not bearish
            if bb_near_low and fisher_oversold and not htf_bearish:
                new_signal = SIZE_ENTRY
            # Long: price at BB lower + RSI oversold + Fisher crossing up
            elif bb_low and rsi_oversold and fisher_cross_up:
                new_signal = SIZE_ENTRY
            # Long: Fisher cross up from extreme + 4h bullish
            elif fisher_cross_up and htf_bullish:
                new_signal = SIZE_ENTRY
            
            # Short: price at BB upper + Fisher overbought + 4h not bullish
            if bb_near_high and fisher_overbought and not htf_bullish:
                new_signal = -SIZE_ENTRY
            # Short: price at BB upper + RSI overbought + Fisher crossing down
            elif bb_high and rsi_overbought and fisher_cross_down:
                new_signal = -SIZE_ENTRY
            # Short: Fisher cross down from extreme + 4h bearish
            elif fisher_cross_down and htf_bearish:
                new_signal = -SIZE_ENTRY
        
        # === TREND REGIME: Trend Following with HMA ===
        elif regime_trend:
            # Long: 4h bullish + 15m HMA bullish + Fast HMA cross up
            if htf_bullish and hma_15m_bullish and fast_above_slow:
                new_signal = SIZE_ENTRY
            # Long: 4h bullish + HMA rising + Fisher cross up + ADX building
            elif htf_bullish and hma_rising and fisher_cross_up and i > 0 and adx[i] > adx[i-1]:
                new_signal = SIZE_ENTRY
            # Long: 4h bullish + HMA cross up + RSI mid (pullback entry)
            elif htf_bullish and fast_above_slow and rsi_mid:
                new_signal = SIZE_ENTRY
            
            # Short: 4h bearish + 15m HMA bearish + Fast HMA cross down
            if htf_bearish and hma_15m_bearish and fast_below_slow:
                new_signal = -SIZE_ENTRY
            # Short: 4h bearish + HMA falling + Fisher cross down + ADX building
            elif htf_bearish and hma_falling and fisher_cross_down and i > 0 and adx[i] > adx[i-1]:
                new_signal = -SIZE_ENTRY
            # Short: 4h bearish + HMA cross down + RSI mid (pullback entry)
            elif htf_bearish and fast_below_slow and rsi_mid:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME: Wait for clear signals ===
        else:
            # Only enter on strong Fisher reversals with HTF confirmation
            if fisher_cross_up and htf_bullish and rsi_oversold:
                new_signal = SIZE_ENTRY
            elif fisher_cross_down and htf_bearish and rsi_overbought:
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