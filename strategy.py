#!/usr/bin/env python3
"""
Experiment #1394: 4h Dual Regime — Choppiness Index Switch + HMA/Donchian/RSI

Hypothesis: Previous 4h failures used单一 regime (either pure trend or pure mean revert).
The research notes show Choppiness Index > 61.8 = range (mean revert), < 38.2 = trend.
This strategy ADAPTS to market regime instead of forcing one approach.

Key insight from #1391: Clean Donchian + HMA worked but Sharpe only 0.305.
Adding regime detection should improve risk-adjusted returns by:
- Mean revert in choppy markets (RSI extremes + BB bands)
- Trend follow in trending markets (HMA + Donchian breakout)
- 12h HMA for macro bias (proven in #1391)

Design:
1. 12h HMA(21) = macro trend bias (from mtf_data, called ONCE)
2. 4h Choppiness Index(14) = regime detector (>55 = chop, <45 = trend)
3. CHOPPY regime: RSI(14)<30 + price<BB_lower → long; RSI>70 + price>BB_upper → short
4. TREND regime: Price>12h_HMA + Donchian(20) breakout → long; opposite for short
5. ATR(14) trailing stop 2.5x = risk management
6. Position size 0.28 = conservative for regime switching
7. Hysteresis on regime: enter at 55/45, exit at 50 to avoid whipsaw

Target: 30-50 trades/year, Sharpe > 0.618 (beat current best), trades >= 30 train
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_hma_donchian_rsi_atr_v1"
timeframe = "4h"
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

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion boundaries"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    mid = sma
    
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return chop

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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Regime hysteresis tracking
    prev_regime = 0  # 0 = unknown, 1 = trend, 2 = chop
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(donchian_20_upper[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION with HYSTERESIS ===
        # Enter trend regime at CHOP < 45, exit at CHOP > 50
        # Enter chop regime at CHOP > 55, exit at CHOP < 50
        current_chop = chop[i]
        
        if prev_regime == 1:  # Was in trend regime
            if current_chop > 50:
                regime = 2  # Switch to chop
            else:
                regime = 1  # Stay in trend
        elif prev_regime == 2:  # Was in chop regime
            if current_chop < 50:
                regime = 1  # Switch to trend
            else:
                regime = 2  # Stay in chop
        else:  # Unknown initial regime
            if current_chop < 45:
                regime = 1
            elif current_chop > 55:
                regime = 2
            else:
                regime = prev_regime if prev_regime != 0 else 1
        
        prev_regime = regime
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === DESIRED SIGNAL based on REGIME ===
        desired_signal = 0.0
        
        if regime == 2:  # CHOPPY REGIME - Mean Reversion
            # LONG: RSI oversold + price at/near BB lower
            if rsi[i] < 35.0 and close[i] <= bb_lower[i] * 1.002:
                if macro_bull:  # Only long if macro bullish
                    desired_signal = BASE_SIZE
                elif not macro_bear:  # Neutral macro OK for mean revert
                    desired_signal = BASE_SIZE * 0.5
            
            # SHORT: RSI overbought + price at/near BB upper
            elif rsi[i] > 65.0 and close[i] >= bb_upper[i] * 0.998:
                if macro_bear:  # Only short if macro bearish
                    desired_signal = -BASE_SIZE
                elif not macro_bull:  # Neutral macro OK for mean revert
                    desired_signal = -BASE_SIZE * 0.5
        
        elif regime == 1:  # TREND REGIME - Trend Following
            # LONG: Price above 12h HMA + Donchian breakout
            if macro_bull and close[i] > donchian_20_upper[i-1]:
                if rsi[i] > 40.0:  # Momentum confirmation
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = BASE_SIZE * 0.5
            
            # SHORT: Price below 12h HMA + Donchian breakdown
            elif macro_bear and close[i] < donchian_20_lower[i-1]:
                if rsi[i] < 60.0:  # Momentum confirmation
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -BASE_SIZE * 0.5
            
            # Trend continuation without breakout
            elif macro_bull and rsi[i] > 50.0 and close[i] > hma_12h_aligned[i] * 1.005:
                desired_signal = BASE_SIZE * 0.5
            elif macro_bear and rsi[i] < 50.0 and close[i] < hma_12h_aligned[i] * 0.995:
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
            if desired_signal > 0:
                final_signal = BASE_SIZE
            else:
                final_signal = -BASE_SIZE
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