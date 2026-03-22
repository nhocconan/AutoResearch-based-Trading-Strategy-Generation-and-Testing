#!/usr/bin/env python3
"""
Experiment #013: 15m Regime-Adaptive Strategy with 4h HMA Bias + Choppiness Index
Hypothesis: 15m timeframe captures more opportunities than 12h/1d while 4h HMA provides
stable trend bias. Choppiness Index (CHOP) detects regime: CHOP>61.8=range (mean revert),
CHOP<38.2=trend (trend follow). This adaptive approach should work in both 2021-2024 bull
and 2025 bear/range markets. Multiple entry paths ensure >=10 trades per symbol.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_chop_4h_hma_bb_rsi_atr_v1"
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
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
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
    
    return np.clip(chop, 0, 100)

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

def calculate_keltner(high, low, close, period=20, multiplier=2.0):
    """Calculate Keltner Channel for squeeze detection."""
    atr = calculate_atr(high, low, close, period)
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    return upper, lower, ema

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
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    kc_upper, kc_lower, kc_mid = calculate_keltner(high, low, close, 20, 2.0)
    
    # 15m HMA for trend
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 10)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        daily_bullish = close[i] > hma_4h_aligned[i]
        daily_bearish = close[i] < hma_4h_aligned[i]
        
        # Regime detection via Choppiness Index
        regime_range = chop[i] > 55  # Range/choppy market
        regime_trend = chop[i] < 45  # Trending market
        
        # 15m HMA trend
        hma_15m_bullish = close[i] > hma_15m[i]
        hma_15m_bearish = close[i] < hma_15m[i]
        hma_rising = hma_15m[i] > hma_15m[i-5] if i > 5 else False
        hma_falling = hma_15m[i] < hma_15m[i-5] if i > 5 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_15m_fast[i] > hma_15m[i]
        fast_below_slow = hma_15m_fast[i] < hma_15m[i]
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 50
        rsi_pullback_short = rsi[i] > 50 and rsi[i] < 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # Bollinger position
        price_near_bb_lower = close[i] < bb_lower[i] * 1.002
        price_near_bb_upper = close[i] > bb_upper[i] * 0.998
        bb_squeeze = (bb_upper[i] - bb_lower[i]) < (kc_upper[i] - kc_lower[i]) * 0.9
        
        new_signal = 0.0
        
        # === TRENDING REGIME ENTRIES (CHOP < 45) ===
        
        if regime_trend:
            # Long: 4h bullish + 15m HMA bullish + RSI pullback
            if daily_bullish and hma_15m_bullish and rsi_pullback_long:
                new_signal = SIZE_ENTRY
            
            # Long: 4h bullish + Fast HMA crossover up + RSI rising
            elif daily_bullish and fast_above_slow and rsi[i] > rsi[i-1] if i > 0 else False:
                new_signal = SIZE_ENTRY
            
            # Short: 4h bearish + 15m HMA bearish + RSI pullback
            elif daily_bearish and hma_15m_bearish and rsi_pullback_short:
                new_signal = -SIZE_ENTRY
            
            # Short: 4h bearish + Fast HMA crossover down + RSI falling
            elif daily_bearish and fast_below_slow and rsi[i] < rsi[i-1] if i > 0 else False:
                new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME ENTRIES (CHOP > 55) ===
        
        if regime_range:
            # Long: Price at BB lower + RSI oversold + 4h not bearish
            if price_near_bb_lower and rsi_oversold and not daily_bearish:
                new_signal = SIZE_ENTRY
            
            # Long: BB squeeze breakout up + RSI neutral
            elif bb_squeeze and close[i] > bb_mid[i] and rsi_neutral:
                new_signal = SIZE_ENTRY
            
            # Short: Price at BB upper + RSI overbought + 4h not bullish
            elif price_near_bb_upper and rsi_overbought and not daily_bullish:
                new_signal = -SIZE_ENTRY
            
            # Short: BB squeeze breakout down + RSI neutral
            elif bb_squeeze and close[i] < bb_mid[i] and rsi_neutral:
                new_signal = -SIZE_ENTRY
        
        # === TRANSITION ZONE (45 <= CHOP <= 55) - Use HMA crossover ===
        
        if not regime_trend and not regime_range:
            # Long: 4h bullish + HMA crossover up
            if daily_bullish and fast_above_slow and hma_15m[i] > hma_15m[i-1] if i > 0 else False:
                new_signal = SIZE_ENTRY
            
            # Short: 4h bearish + HMA crossover down
            elif daily_bearish and fast_below_slow and hma_15m[i] < hma_15m[i-1] if i > 0 else False:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
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
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
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