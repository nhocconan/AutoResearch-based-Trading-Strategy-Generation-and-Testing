#!/usr/bin/env python3
"""
Experiment #474: 4h Primary + 12h/1d HTF — HMA Trend + Donchian Breakout + ADX Filter + Choppiness Regime

Hypothesis: Based on successful patterns from experiment history, HMA + Donchian + RSI worked well
on 4h timeframe. Key innovations for this version:
1. HMA (Hull MA) - faster response than EMA/KAMA, less lag for trend detection
2. Donchian Channel (20) - breakout signals with proven edge on 4h
3. ADX (14) - trend strength filter, only trade when ADX > 20 (avoid choppy whipsaws)
4. Choppiness Index (14) - regime detection: >55 = mean revert, <45 = trend follow
5. RSI (14) - entry timing filter, avoid extreme overbought/oversold on entries
6. 12h HMA + 1d HMA for HTF bias alignment (call get_htf_data ONCE before loop)
7. ATR(14) trailing stop at 2.0x for tighter risk management
8. Discrete position sizing: 0.0, ±0.25, ±0.35 to minimize fee churn

Why this should work: HMA is faster than KAMA (failed in #469, #471, #472). Donchian breakouts
worked in best strategy (mtf_4h_triple_regime_crsi_donchian_1d1w_v1 Sharpe=0.612). ADX filter
prevents trades in weak trends (major source of losses). 4h TF targets 20-50 trades/year.
Target: Sharpe > 0.612, DD < -35%, trades >= 30 on train, >= 3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_adx_chop_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average - faster response, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    if half < 1 or sqrt_period < 1:
        return hma
    
    # WMA calculation helper
    def wma(data, span):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    diff = 2.0 * wma_half - wma_full
    
    for i in range(sqrt_period - 1, n):
        if np.isnan(diff[i - sqrt_period + 1:i + 1]).any():
            continue
        window = diff[i - sqrt_period + 1:i + 1]
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.sum(window * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth with Wilder's method
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
        minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
        
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP). High = choppy, Low = trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        atr_avg = atr_sum / period
        
        if highest - lowest > 1e-10 and atr_avg > 1e-10:
            chop[i] = 100.0 * np.log10((highest - lowest) / (atr_avg * np.sqrt(period))) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    rsi_14 = calculate_rsi(close, 14)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    chop = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.35
    SIZE_SHORT = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(chop[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market - mean revert
        is_trending = chop[i] < 45.0  # Trend market - breakout
        
        # === PRIMARY TREND (HMA crossover) ===
        trend_bullish = hma_16[i] > hma_48[i]
        trend_bearish = hma_16[i] < hma_48[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_14[i] > 20.0  # Only trade when trend has strength
        weak_trend = adx_14[i] <= 20.0
        
        # === HTF TREND BIAS (12h + 1d HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === RSI FILTER ===
        rsi_not_overbought = rsi_14[i] < 70.0  # Avoid buying at extremes
        rsi_not_oversold = rsi_14[i] > 30.0  # Avoid selling at extremes
        rsi_bullish = rsi_14[i] > 50.0
        rsi_bearish = rsi_14[i] < 50.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        long_score = 0
        
        # Trend following mode (trending market + strong ADX)
        if is_trending and strong_trend and trend_bullish:
            long_score += 2
            if breakout_long:
                long_score += 2  # Breakout confirmation
            if rsi_not_overbought and rsi_bullish:
                long_score += 1
        
        # Mean reversion mode (choppy market)
        if is_choppy and trend_bullish:
            long_score += 1
            if rsi_14[i] < 40.0:  # Oversold in uptrend
                long_score += 2
        
        # HTF alignment bonus
        if price_above_hma_12h:
            long_score += 1
        if price_above_hma_1d:
            long_score += 1
        
        # DI confirmation
        if plus_di[i] > minus_di[i]:
            long_score += 1
        
        if long_score >= 4:
            desired_signal = SIZE_LONG
        
        # SHORT ENTRIES
        if desired_signal == 0.0:
            short_score = 0
            
            # Trend following mode (trending market + strong ADX)
            if is_trending and strong_trend and trend_bearish:
                short_score += 2
                if breakout_short:
                    short_score += 2  # Breakout confirmation
                if rsi_not_oversold and rsi_bearish:
                    short_score += 1
            
            # Mean reversion mode (choppy market)
            if is_choppy and trend_bearish:
                short_score += 1
                if rsi_14[i] > 60.0:  # Overbought in downtrend
                    short_score += 2
            
            # HTF alignment bonus
            if price_below_hma_12h:
                short_score += 1
            if price_below_hma_1d:
                short_score += 1
            
            # DI confirmation
            if minus_di[i] > plus_di[i]:
                short_score += 1
            
            if short_score >= 4:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and trend_bullish and price_above_hma_12h:
                desired_signal = SIZE_LONG
            elif position_side < 0 and trend_bearish and price_below_hma_12h:
                desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = 0.35
        elif desired_signal < 0:
            desired_signal = -0.30
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals