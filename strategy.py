#!/usr/bin/env python3
"""
Experiment #1234: 4h Primary + 12h HTF — Donchian Breakout with Choppiness Regime Filter

Hypothesis: Previous KAMA+ADX strategies (#1229) worked but didn't beat the best (Sharpe 0.612).
Research shows Donchian breakouts work well for SOL (Sharpe +0.782), while Choppiness Index
regime detection works for ETH (Sharpe +0.923). This combines both: use Choppiness to detect
whether we're in trend or range, then apply appropriate entry logic.

Key innovations:
1. Choppiness Index (CHOP) regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
2. In trend regime: Donchian(20) breakout + 12h HMA bias
3. In range regime: RSI extremes + Bollinger mean reversion
4. 12h HMA for macro bias (proven in #1222, #1226)
5. Fewer conflicting conditions = more trades while maintaining quality

Target: Sharpe > 0.612, trades >= 80 train (20/year), >= 12 test (3/year), DD > -50%
Timeframe: 4h (20-50 trades/year target)
Position Size: 0.28 (discrete: 0.0, ±0.28)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_chop_regime_12h_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate CHOP
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = Highest High over period
    Lower = Lowest Low over period
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_bb(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return middle, upper, lower
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return middle, upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bb(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(bb_mid[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        in_trend_regime = chop[i] < 38.2  # Trending market
        in_range_regime = chop[i] > 61.8  # Range/choppy market
        # Neutral regime: 38.2 <= CHOP <= 61.8
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # TREND REGIME: Donchian breakout with macro bias
        if in_trend_regime:
            # Long: Price breaks Donchian upper + macro bull
            if close[i] > donchian_upper[i - 1] and macro_bull:
                desired_signal = BASE_SIZE
            
            # Short: Price breaks Donchian lower + macro bear
            elif close[i] < donchian_lower[i - 1] and macro_bear:
                desired_signal = -BASE_SIZE
        
        # RANGE REGIME: Mean reversion at Bollinger bounds
        elif in_range_regime:
            # Long: Price at BB lower + RSI oversold
            if close[i] <= bb_lower[i] and rsi[i] < 35.0:
                desired_signal = BASE_SIZE
            
            # Short: Price at BB upper + RSI overbought
            elif close[i] >= bb_upper[i] and rsi[i] > 65.0:
                desired_signal = -BASE_SIZE
        
        # NEUTRAL REGIME: Only trade strong breakouts with RSI confirmation
        else:
            # Long: Donchian breakout + RSI momentum + macro bull
            if close[i] > donchian_upper[i - 1] and rsi[i] > 55.0 and macro_bull:
                desired_signal = BASE_SIZE
            
            # Short: Donchian breakout + RSI momentum + macro bear
            elif close[i] < donchian_lower[i - 1] and rsi[i] < 45.0 and macro_bear:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (ATR-based 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Trailing stop for long: highest since entry - 2.5*ATR
            current_stop = entry_price + 2.5 * atr[i] if entry_price > 0 else stop_price
            # Update trailing stop upward only
            if close[i] > entry_price:
                stop_price = max(stop_price, close[i] - 2.5 * atr[i])
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Trailing stop for short: lowest since entry + 2.5*ATR
            if close[i] < entry_price:
                stop_price = min(stop_price, close[i] + 2.5 * atr[i])
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                stop_price = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                stop_price = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals