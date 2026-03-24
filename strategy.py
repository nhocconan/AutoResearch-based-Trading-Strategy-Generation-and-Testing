#!/usr/bin/env python3
"""
Experiment #691: 6h Primary + 1d/1w HTF — KAMA Adaptive Trend + StochRSI Entry + ADX Filter

Hypothesis: 6h timeframe captures multi-day swings better than 4h (less noise) and 12h (faster response).
Using KAMA (Kaufman Adaptive Moving Average) instead of HMA/EMA - KAMA adapts to market efficiency,
moving fast in trends and slow in chop. This should reduce whipsaw during 2022 crash while capturing
2021 bull run. StochRSI for precise entry timing (less lag than regular RSI). ADX filter ensures we
only trade when trend has sufficient strength.

Key innovations:
1. KAMA(10,2,30) - adapts smoothing based on volatility (ER = efficiency ratio)
2. StochRSI(14,3,3) - more sensitive than RSI for entry timing, catches turns earlier
3. ADX(14) > 18 filter - only trade when trend has momentum (avoids chop)
4. 1d KAMA(21) direction bias - primary HTF filter
5. 1w KAMA(21) meta-filter - avoid counter-trend against weekly
6. Asymmetric sizing: 0.25 base, 0.35 when all HTF align strongly
7. ATR(14) 2.5x trailing stop for risk management

Entry conditions (balanced for trade generation):
- LONG: price > 1d KAMA AND 1d KAMA rising AND 6h KAMA > KAMA_slow AND StochRSI K crosses above D
        AND StochRSI < 0.7 (not overbought) AND ADX > 18
- SHORT: price < 1d KAMA AND 1d KAMA falling AND 6h KAMA < KAMA_slow AND StochRSI K crosses below D
         AND StochRSI > 0.3 (not oversold) AND ADX > 18

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.35 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_adaptive_stochrsi_adx_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, fast_period=10, slow_period=30, smoothing_period=2):
    """
    Kaufman Adaptive Moving Average - adapts to market efficiency
    Fast period: 10, Slow period: 30, Smoothing: 2 (standard settings)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < slow_period + 1:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(slow_period, n):
        price_change = abs(close[i] - close[i - slow_period])
        if price_change < 1e-10:
            er[i] = 0.0
        else:
            volatility = np.sum(np.abs(np.diff(close[i - slow_period:i + 1])))
            if volatility < 1e-10:
                er[i] = 0.0
            else:
                er[i] = price_change / volatility
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_stoch_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """
    Stochastic RSI - more sensitive than regular RSI
    Returns StochRSI K and D lines
    """
    n = len(close)
    stoch_k = np.full(n, np.nan)
    stoch_d = np.full(n, np.nan)
    
    if n < rsi_period + stoch_period + d_period:
        return stoch_k, stoch_d
    
    # First calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] > 1e-10:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi / 100.0  # Normalize to 0-1
    
    # Calculate Stochastic of RSI
    for i in range(rsi_period + stoch_period - 1, n):
        rsi_low = np.min(rsi[i - stoch_period + 1:i + 1])
        rsi_high = np.max(rsi[i - stoch_period + 1:i + 1])
        
        if rsi_high - rsi_low < 1e-10:
            stoch_k[i] = 0.5
        else:
            stoch_k[i] = (rsi[i] - rsi_low) / (rsi_high - rsi_low)
    
    # Smooth K to get D
    for i in range(rsi_period + stoch_period + k_period - 2, n):
        if not np.isnan(stoch_k[i - k_period + 1:i + 1]).all():
            stoch_d[i] = np.nanmean(stoch_k[i - k_period + 1:i + 1])
    
    return stoch_k, stoch_d

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 3:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth TR and DM
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period * 2 - 1:] = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values[period * 2 - 1:]
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF KAMA
    kama_1d_raw = calculate_kama(df_1d['close'].values, fast_period=10, slow_period=30, smoothing_period=2)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, fast_period=10, slow_period=30, smoothing_period=2)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate 6h indicators
    kama_fast = calculate_kama(close, fast_period=10, slow_period=30, smoothing_period=2)
    kama_slow = calculate_kama(close, fast_period=20, slow_period=50, smoothing_period=2)
    stoch_k, stoch_d = calculate_stoch_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
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
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx[i]) or np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w KAMA) ===
        htf_1d_bull = close[i] > kama_1d_aligned[i]
        htf_1d_bear = close[i] < kama_1d_aligned[i]
        
        # Check 1d KAMA direction (rising/falling)
        htf_1d_rising = False
        htf_1d_falling = False
        if i >= 2 and not np.isnan(kama_1d_aligned[i-1]):
            htf_1d_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1]
            htf_1d_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1]
        
        htf_1w_bull = close[i] > kama_1w_aligned[i]
        htf_1w_bear = close[i] < kama_1w_aligned[i]
        
        # === KAMA CROSSOVER TREND ===
        kama_bull = kama_fast[i] > kama_slow[i]
        kama_bear = kama_fast[i] < kama_slow[i]
        
        # === STOCHRSI ENTRY SIGNAL ===
        # Check for K crossing above D (bullish) or below D (bearish)
        stoch_cross_bull = False
        stoch_cross_bear = False
        
        if i >= 2 and not np.isnan(stoch_k[i-1]) and not np.isnan(stoch_d[i-1]):
            # K crosses above D
            if stoch_k[i-1] <= stoch_d[i-1] and stoch_k[i] > stoch_d[i]:
                stoch_cross_bull = True
            # K crosses below D
            if stoch_k[i-1] >= stoch_d[i-1] and stoch_k[i] < stoch_d[i]:
                stoch_cross_bear = True
        
        # StochRSI not at extremes
        stoch_not_overbought = stoch_k[i] < 0.75
        stoch_not_oversold = stoch_k[i] > 0.25
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx[i] > 18.0  # Trend has sufficient strength
        
        # === ENTRY LOGIC (BALANCED CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG: HTF bullish + KAMA bull + StochRSI cross + ADX strong
        if htf_1d_bull and htf_1d_rising and kama_bull and stoch_cross_bull and stoch_not_overbought and adx_strong:
            if htf_1w_bull:
                # All HTF aligned - strong signal
                desired_signal = SIZE_STRONG
            else:
                # Just 1d aligned - base signal
                desired_signal = SIZE_BASE
        elif htf_1d_bull and kama_bull and stoch_cross_bull and adx_strong:
            # Weaker: just 1d bias + KAMA + StochRSI
            desired_signal = SIZE_BASE * 0.75
        
        # SHORT: HTF bearish + KAMA bear + StochRSI cross + ADX strong
        elif htf_1d_bear and htf_1d_falling and kama_bear and stoch_cross_bear and stoch_not_oversold and adx_strong:
            if htf_1w_bear:
                # All HTF aligned - strong signal
                desired_signal = -SIZE_STRONG
            else:
                # Just 1d aligned - base signal
                desired_signal = -SIZE_BASE
        elif htf_1d_bear and kama_bear and stoch_cross_bear and adx_strong:
            # Weaker: just 1d bias + KAMA + StochRSI
            desired_signal = -SIZE_BASE * 0.75
        
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
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.75
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