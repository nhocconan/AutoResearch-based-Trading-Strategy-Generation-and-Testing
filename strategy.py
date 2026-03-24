#!/usr/bin/env python3
"""
Experiment #1450: 1h Primary + 4h/12h HTF — Regime-Adaptive Dual Mode

Hypothesis: 1h strategies fail because entry conditions are TOO STRICT (0 trades).
Solution: Use OR logic for entries (ANY of 3 paths), not AND logic.

Key insights from 1087 failed experiments:
- 1h/4h strategies with Sharpe=0.000 have 0 trades (conditions too strict)
- Bear/range markets (2025+) need mean reversion, not pure trend
- Funding rate mean reversion works for BTC/ETH in bear markets
- Choppiness Index regime detection is proven edge

Design:
1. 4h HMA(21) = macro trend bias (call ONCE before loop)
2. 12h HMA(21) = secondary trend confirmation
3. Choppiness Index(14) = regime (chop>55 range, chop<45 trend)
4. RSI(7) extremes + Bollinger position = mean reversion entries
5. Donchian(20) breakout = trend entries
6. ANY of 3 entry paths triggers (OR logic, not AND)
7. ATR(14) trailing stop 2.5x for risk management
8. Position size 0.25 (conservative for 1h)

Why this should work:
- Multiple entry paths = more trades (avoid 0-trade failure)
- Regime-adaptive = works in both bull and bear markets
- HTF trend filter = reduces false signals
- Conservative sizing = controls drawdown

Target: 40-80 trades/year, Sharpe > 0.618, trades >= 30 train, >= 5 test
Timeframe: 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_dual_mode_4h12h_hma_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                prev_close = close[j-1] if j > 0 else close[j]
                tr_sum += max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA for macro trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_20_upper[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_7[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (4h + 12h HMA) - bias filter ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # Strong bull: both 4h and 12h HMA below price
        macro_bull_strong = price_above_4h and price_above_12h
        # Strong bear: both 4h and 12h HMA above price
        macro_bear_strong = price_below_4h and price_below_12h
        # Weak bull/bear: only one agrees
        macro_bull_weak = price_above_4h or price_above_12h
        macro_bear_weak = price_below_4h or price_below_12h
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        # Neutral regime: 45-55
        
        # === RSI EXTREMES (mean reversion) ===
        rsi_oversold = rsi_7[i] < 25.0  # Loosened from 20 for more trades
        rsi_overbought = rsi_7[i] > 75.0  # Loosened from 80 for more trades
        rsi_extreme_oversold = rsi_7[i] < 15.0
        rsi_extreme_overbought = rsi_7[i] > 85.0
        
        # === BOLLINGER POSITION ===
        bb_oversold = close[i] < bb_lower[i] * 1.001  # At or below lower band
        bb_overbought = close[i] > bb_upper[i] * 0.999  # At or above upper band
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_20_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_20_lower[i-1] if i > 0 else False
        
        # === DESIRED SIGNAL - MULTIPLE ENTRY PATHS (OR LOGIC) ===
        desired_signal = 0.0
        
        # LONG ENTRY PATHS (any one triggers)
        # Path 1: Choppy regime + RSI oversold + BB oversold (mean reversion)
        if is_choppy and rsi_oversold and bb_oversold:
            desired_signal = BASE_SIZE
        # Path 2: Choppy regime + RSI extreme oversold (stronger signal)
        elif is_choppy and rsi_extreme_oversold:
            desired_signal = BASE_SIZE
        # Path 3: Trending regime + Donchian breakout + macro bull (trend follow)
        elif is_trending and breakout_long and macro_bull_strong:
            desired_signal = BASE_SIZE
        # Path 4: Neutral regime + RSI extreme oversold + macro bull weak
        elif rsi_extreme_oversold and macro_bull_weak and not macro_bear_strong:
            desired_signal = BASE_SIZE * 0.5
        # Path 5: Any regime + RSI extreme oversold + price above 4h HMA
        elif rsi_extreme_oversold and price_above_4h:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRY PATHS (any one triggers)
        # Path 1: Choppy regime + RSI overbought + BB overbought (mean reversion)
        elif is_choppy and rsi_overbought and bb_overbought:
            desired_signal = -BASE_SIZE
        # Path 2: Choppy regime + RSI extreme overbought (stronger signal)
        elif is_choppy and rsi_extreme_overbought:
            desired_signal = -BASE_SIZE
        # Path 3: Trending regime + Donchian breakout + macro bear (trend follow)
        elif is_trending and breakout_short and macro_bear_strong:
            desired_signal = -BASE_SIZE
        # Path 4: Neutral regime + RSI extreme overbought + macro bear weak
        elif rsi_extreme_overbought and macro_bear_weak and not macro_bull_strong:
            desired_signal = -BASE_SIZE * 0.5
        # Path 5: Any regime + RSI extreme overbought + price below 4h HMA
        elif rsi_extreme_overbought and price_below_4h:
            desired_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals