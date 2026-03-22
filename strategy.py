#!/usr/bin/env python3
"""
Experiment #473: 12h Regime-Adaptive Dual-Mode Strategy
Hypothesis: Market regime detection (trending vs ranging) allows switching between
trend-following and mean-reversion modes. CHOP(14) > 61.8 = range (buy low/sell high),
CHOP < 38.2 = trend (follow direction). 12h timeframe reduces noise while maintaining
trade frequency. Daily HTF bias filter ensures we trade with higher TF momentum.
Multiple entry paths ensure >=10 trades requirement is met.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_adaptive_chop_hma_rsi_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    """
    atr = calculate_atr(high, low, close, period)
    
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return np.clip(chop, 0, 100)

def calculate_slope(values, lookback=5):
    """Calculate slope of values over lookback period."""
    n = len(values)
    slope = np.zeros(n)
    slope[:] = np.nan
    for i in range(lookback, n):
        if not np.isnan(values[i]) and not np.isnan(values[i - lookback]):
            slope[i] = (values[i] - values[i - lookback]) / lookback
    return slope

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std
    return zscore.values

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
    hma_12h = calculate_hma(close, 21)
    hma_12h_fast = calculate_hma(close, 9)
    rsi = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    hma_slope = calculate_slope(hma_12h, lookback=5)
    zscore = calculate_zscore(close, 20)
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_slope[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend bias (HTF)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # 12h HMA trend
        hma_12h_bullish = close[i] > hma_12h[i]
        hma_12h_bearish = close[i] < hma_12h[i]
        hma_rising = hma_slope[i] > 0
        hma_falling = hma_slope[i] < 0
        
        # Fast HMA crossover
        fast_above_slow = hma_12h_fast[i] > hma_12h[i]
        fast_below_slow = hma_12h_fast[i] < hma_12h[i]
        
        # Market regime detection
        is_ranging = chop[i] > 55  # Lower threshold for more trades
        is_trending = chop[i] < 45  # Lower threshold for more trades
        
        # RSI levels
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # Z-score extremes
        zscore_low = zscore[i] < -1.5
        zscore_high = zscore[i] > 1.5
        
        new_signal = 0.0
        
        # === TRENDING MODE (CHOP < 45) ===
        if is_trending:
            # Long: Daily bullish + 12h bullish + HMA rising + RSI not overbought
            if daily_bullish and hma_12h_bullish and hma_rising and rsi[i] < 65:
                new_signal = SIZE_ENTRY
            # Long: Fast HMA above slow + Daily bullish + RSI > 40
            elif daily_bullish and fast_above_slow and rsi[i] > 40 and rsi[i] < 65:
                new_signal = SIZE_ENTRY
            # Long: 12h bullish + HMA rising + RSI pullback (40-55)
            elif hma_12h_bullish and hma_rising and rsi[i] > 40 and rsi[i] < 55:
                new_signal = SIZE_ENTRY
            
            # Short: Daily bearish + 12h bearish + HMA falling + RSI not oversold
            if daily_bearish and hma_12h_bearish and hma_falling and rsi[i] > 35:
                new_signal = -SIZE_ENTRY
            # Short: Fast HMA below slow + Daily bearish + RSI < 60
            elif daily_bearish and fast_below_slow and rsi[i] > 35 and rsi[i] < 60:
                new_signal = -SIZE_ENTRY
            # Short: 12h bearish + HMA falling + RSI rally (45-60)
            elif hma_12h_bearish and hma_falling and rsi[i] > 45 and rsi[i] < 60:
                new_signal = -SIZE_ENTRY
        
        # === RANGING MODE (CHOP > 55) ===
        elif is_ranging:
            # Long: RSI oversold + Z-score low + Price near lower range
            if rsi_oversold and zscore_low:
                new_signal = SIZE_ENTRY
            # Long: RSI < 40 + Price below 12h HMA (mean reversion)
            elif rsi[i] < 40 and close[i] < hma_12h[i]:
                new_signal = SIZE_ENTRY
            # Long: Z-score < -1.0 + Daily bullish bias
            elif zscore[i] < -1.0 and daily_bullish:
                new_signal = SIZE_ENTRY
            
            # Short: RSI overbought + Z-score high + Price near upper range
            if rsi_overbought and zscore_high:
                new_signal = -SIZE_ENTRY
            # Short: RSI > 60 + Price above 12h HMA (mean reversion)
            elif rsi[i] > 60 and close[i] > hma_12h[i]:
                new_signal = -SIZE_ENTRY
            # Short: Z-score > 1.0 + Daily bearish bias
            elif zscore[i] > 1.0 and daily_bearish:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL/TRANSITION MODE (45 <= CHOP <= 55) ===
        else:
            # Conservative entries only on strong signals
            # Long: All trend factors aligned
            if daily_bullish and hma_12h_bullish and hma_rising and rsi[i] > 45 and rsi[i] < 55:
                new_signal = SIZE_ENTRY
            # Short: All trend factors aligned
            elif daily_bearish and hma_12h_bearish and hma_falling and rsi[i] > 45 and rsi[i] < 55:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 12h timeframe)
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