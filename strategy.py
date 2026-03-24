#!/usr/bin/env python3
"""
Experiment #888: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + ADX + Choppiness Regime

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency -
slower in choppy markets, faster in trends. Combined with ADX for trend strength
and Choppiness Index for regime detection, this should work better than static
HMA/EMA in bear/range markets (2025 test period).

Key innovations:
1. KAMA(14) with ER-based adaptation - proven to reduce whipsaw in ranges
2. ADX(14) > 25 for trend confirmation, < 20 for range detection
3. Choppiness(14) regime switch: >55 = mean revert, <45 = trend follow
4. 12h KAMA for HTF bias, 1d KAMA for macro trend
5. ATR(14) 2.5x trailing stop
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE for trades):
- TREND REGIME (ADX>25, CHOP<45): LONG = 12h KAMA bull + price>KAMA(14)
- TREND REGIME (ADX>25, CHOP<45): SHORT = 12h KAMA bear + price<KAMA(14)
- RANGE REGIME (ADX<20, CHOP>55): LONG = 12h KAMA bull + RSI(14)<35
- RANGE REGIME (ADX<20, CHOP>55): SHORT = 12h KAMA bear + RSI(14)>65

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_chop_regime_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market efficiency ratio (ER)
    ER = |change| / sum(|change|) over period
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prior_KAMA + SC * (price - prior_KAMA)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Efficiency Ratio
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_changes > 1e-10:
            er[i] = change / sum_changes
        else:
            er[i] = 0.0
    
    # Smoothing Constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Initialize KAMA at SMA
    kama[period] = np.mean(close[:period + 1])
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            continue
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100.0 * minus_dm_s[i] / tr_s[i]
    
    # DX
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 45/55 thresholds for regime switch
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMA
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=14)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 4h indicators
    kama_14 = calculate_kama(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_14[i]) or np.isnan(adx_14[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h/1d KAMA) ===
        htf_12h_bull = close[i] > kama_12h_aligned[i]
        htf_12h_bear = close[i] < kama_12h_aligned[i]
        htf_1d_bull = close[i] > kama_1d_aligned[i]
        htf_1d_bear = close[i] < kama_1d_aligned[i]
        
        # Require both 12h and 1d aligned for strong bias
        htf_bull = htf_12h_bull and htf_1d_bull
        htf_bear = htf_12h_bear and htf_1d_bear
        
        # === ADX TREND STRENGTH ===
        adx_trending = adx_14[i] > 25.0
        adx_ranging = adx_14[i] < 20.0
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 45.0
        chop_ranging = chop_14[i] > 55.0
        
        # === KAMA TREND ===
        kama_bull = close[i] > kama_14[i]
        kama_bear = close[i] < kama_14[i]
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === REGIME DETECTION ===
        is_trend_regime = adx_trending and chop_trending
        is_range_regime = adx_ranging and chop_ranging
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE FOR TRADES) ===
        desired_signal = 0.0
        
        if htf_bull:
            # Bullish HTF bias - prefer longs
            if is_trend_regime:
                # Trend regime: follow KAMA direction
                if kama_bull:
                    desired_signal = SIZE_BASE
                # Strong entry on pullback
                if kama_bull and rsi_oversold:
                    desired_signal = SIZE_STRONG
            elif is_range_regime:
                # Range regime: mean revert on RSI extremes
                if rsi_oversold:
                    desired_signal = SIZE_STRONG
                elif rsi_14[i] < 45.0:
                    desired_signal = SIZE_BASE
            else:
                # Neutral regime: use KAMA with loose RSI
                if kama_bull and rsi_14[i] < 50.0:
                    desired_signal = SIZE_BASE
        
        elif htf_bear:
            # Bearish HTF bias - prefer shorts
            if is_trend_regime:
                # Trend regime: follow KAMA direction
                if kama_bear:
                    desired_signal = -SIZE_BASE
                # Strong entry on pullback
                if kama_bear and rsi_overbought:
                    desired_signal = -SIZE_STRONG
            elif is_range_regime:
                # Range regime: mean revert on RSI extremes
                if rsi_overbought:
                    desired_signal = -SIZE_STRONG
                elif rsi_14[i] > 55.0:
                    desired_signal = -SIZE_BASE
            else:
                # Neutral regime: use KAMA with loose RSI
                if kama_bear and rsi_14[i] > 50.0:
                    desired_signal = -SIZE_BASE
        else:
            # No clear HTF bias - use local signals only (looser)
            if is_range_regime:
                if rsi_oversold:
                    desired_signal = SIZE_BASE
                elif rsi_overbought:
                    desired_signal = -SIZE_BASE
            elif kama_bull and rsi_14[i] < 45.0:
                desired_signal = SIZE_BASE
            elif kama_bear and rsi_14[i] > 55.0:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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