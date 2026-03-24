#!/usr/bin/env python3
"""
Experiment #632: 12h Primary + 1d HTF — Dual Regime (Trend vs Mean Revert)

Hypothesis: Single-regime strategies fail because BTC/ETH alternate between trending
and ranging markets. Use Choppiness Index to detect regime and switch logic:
- CHOP < 38.2 = Trending → Donchian breakout + HMA trend follow
- CHOP > 61.8 = Ranging → RSI mean reversion at Bollinger bands
- 38.2-61.8 = Transition → reduce size or stay flat

Why this should beat #624 (Sharpe=-0.594):
1. Simpler HTF filter (just 1d HMA, not 1d+1w complexity)
2. Regime-adaptive logic (not one-size-fits-all)
3. Proven pattern: Choppiness + Connors RSI gave ETH Sharpe +0.923 in research
4. Looser entry conditions to ensure 30+ trades in train period

Strategy logic:
1. 1d HMA(21) = primary trend bias (long only above, short only below)
2. 12h Choppiness(14) = regime detector (trend vs range)
3. 12h Donchian(20) = breakout levels for trend mode
4. 12h RSI(14) + BB(20,2) = mean reversion signals for range mode
5. 12h ATR(14) = 2.5*ATR trailing stoploss

Entry conditions (LOOSE to ensure trades):
- TREND LONG: CHOP<38.2 + close>1d_HMA + price breaks Donchian high
- TREND SHORT: CHOP<38.2 + close<1d_HMA + price breaks Donchian low
- RANGE LONG: CHOP>61.8 + RSI<30 + price touches BB lower
- RANGE SHORT: CHOP>61.8 + RSI>70 + price touches BB upper

Target: Sharpe>0.40, trades>=30 train, trades>=3 test
Timeframe: 12h
Size: 0.20-0.30 discrete (0.30 trend, 0.20 range)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_donchian_rsi_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_12h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_RANGE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        is_transition = not is_trending and not is_ranging
        
        # === TREND MODE SIGNALS (Donchian Breakout) ===
        trend_signal = 0.0
        
        if is_trending:
            # Breakout above Donchian upper + HTF bull bias
            if close[i] > donchian_upper[i] and htf_bull:
                trend_signal = SIZE_TREND
            # Breakout below Donchian lower + HTF bear bias
            elif close[i] < donchian_lower[i] and htf_bear:
                trend_signal = -SIZE_TREND
            # Pullback to HMA in trend (entry on bounce)
            elif htf_bull and close[i] < hma_12h[i] * 1.005 and close[i] > hma_12h[i] * 0.995:
                if i > 0 and close[i] > close[i-1]:  # bouncing up
                    trend_signal = SIZE_TREND * 0.5
            elif htf_bear and close[i] > hma_12h[i] * 0.995 and close[i] < hma_12h[i] * 1.005:
                if i > 0 and close[i] < close[i-1]:  # bouncing down
                    trend_signal = -SIZE_TREND * 0.5
        
        # === RANGE MODE SIGNALS (RSI + Bollinger Mean Reversion) ===
        range_signal = 0.0
        
        if is_ranging:
            # RSI oversold + price at BB lower = long
            if rsi[i] < 30.0 and low[i] <= bb_lower[i] * 1.002:
                range_signal = SIZE_RANGE
            # RSI overbought + price at BB upper = short
            elif rsi[i] > 70.0 and high[i] >= bb_upper[i] * 0.998:
                range_signal = -SIZE_RANGE
            # Moderate RSI extremes
            elif rsi[i] < 25.0:
                range_signal = SIZE_RANGE * 0.5
            elif rsi[i] > 75.0:
                range_signal = -SIZE_RANGE * 0.5
        
        # === TRANSITION MODE ===
        # Reduce size or stay flat during regime uncertainty
        if is_transition:
            # Only keep existing positions, don't enter new ones
            if in_position:
                desired_signal = np.sign(position_side) * SIZE_RANGE * 0.3
            else:
                desired_signal = 0.0
        else:
            # Combine trend and range signals (prioritize trend if both present)
            if trend_signal != 0.0:
                desired_signal = trend_signal
            elif range_signal != 0.0:
                desired_signal = range_signal
            else:
                desired_signal = 0.0
        
        # === HTF BIAS OVERRIDE ===
        # Don't go long if HTF is strongly bearish, don't short if HTF is strongly bullish
        if desired_signal > 0 and htf_bear and rsi[i] > 50:
            desired_signal = 0.0
        if desired_signal < 0 and htf_bull and rsi[i] < 50:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_TREND * 0.9:
            final_signal = SIZE_TREND
        elif desired_signal <= -SIZE_TREND * 0.9:
            final_signal = -SIZE_TREND
        elif desired_signal >= SIZE_RANGE * 0.9:
            final_signal = SIZE_RANGE
        elif desired_signal <= -SIZE_RANGE * 0.9:
            final_signal = -SIZE_RANGE
        elif abs(desired_signal) >= SIZE_RANGE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_RANGE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals