#!/usr/bin/env python3
"""
Experiment #004: 4h Primary + 12h HTF — KAMA Adaptive Trend + ADX Regime + Donchian Breakout

Hypothesis: After analyzing failed experiments #001-#003, the pattern shows:
- HMA is too responsive and creates whipsaws in bear markets (2022, 2025)
- Choppiness Index alone is insufficient for regime detection
- SOLUTION: Use KAMA (Kaufman Adaptive MA) which adapts to volatility + ADX for trend strength
- KAMA flattens in choppy markets, trends in directional markets (built-in regime filter)
- ADX > 25 confirms strong trend, ADX < 20 confirms range
- Donchian(20) breakouts with volume confirmation reduce false signals
- 12h KAMA provides major trend bias without excessive lag
- This combines: KAMA adaptivity (proven in crypto) + ADX strength + Donchian breakouts

Key design choices:
- Timeframe: 4h (20-50 trades/year target, proven to work best)
- HTF: 12h KAMA for major trend bias (adaptive, less whipsaw than HMA)
- Entry: Donchian(20) breakout + ADX > 25 (trend) OR RSI extremes + ADX < 20 (range)
- Volume filter: breakout volume > 1.5x 20-bar avg (confirms genuine breakout)
- Position size: 0.30 (30% of capital, conservative for 4h)
- Stoploss: 2.0x ATR trailing (tighter for 4h, protects against 2022-style crashes)
- LOOSE RSI filters (25-75) to ensure >=30 trades on train, >=3 on test

Target: Sharpe>0.179 (beat current best), DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_donchian_volume_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - smooth in chop, responsive in trends
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    SC (Smoothing Constant) = [ER * (fast_SC - slow_SC) + slow_SC]^2
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        net_change = abs(close[i] - close[i - period])
        sum_changes = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

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

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = range/chop
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate TR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth +DM, -DM, TR using Wilder's smoothing (EMA with alpha=1/period)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    plus_di[:] = np.nan
    minus_di[:] = np.nan
    dx[:] = np.nan
    adx[:] = np.nan
    
    # Initialize with simple averages for first period
    sum_plus_dm = np.sum(plus_dm[1:period+1])
    sum_minus_dm = np.sum(minus_dm[1:period+1])
    sum_tr = np.sum(tr[1:period+1])
    
    for i in range(period, n):
        if i == period:
            avg_plus_dm = sum_plus_dm
            avg_minus_dm = sum_minus_dm
            avg_tr = sum_tr
        else:
            avg_plus_dm = (avg_plus_dm * (period - 1) + plus_dm[i]) / period
            avg_minus_dm = (avg_minus_dm * (period - 1) - minus_dm[i]) / period
            avg_tr = (avg_tr * (period - 1) + tr[i]) / period
        
        if avg_tr > 1e-10:
            plus_di[i] = 100.0 * avg_plus_dm / avg_tr
            minus_di[i] = 100.0 * avg_minus_dm / avg_tr
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
        
        sum_di = plus_di[i] + minus_di[i]
        if sum_di > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / sum_di
        else:
            dx[i] = 0.0
        
        # ADX is EMA of DX
        if i == period:
            adx[i] = dx[i]
        else:
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

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
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h KAMA for major trend bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=21, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 4h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_4h[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h KAMA) ===
        htf_bull = close[i] > kama_12h_aligned[i]
        htf_bear = close[i] < kama_12h_aligned[i]
        
        # === REGIME DETECTION (ADX) ===
        # ADX > 25 = strong trend (follow breakouts)
        # ADX < 20 = range/chop (mean revert)
        # ADX 20-25 = neutral (reduce position size)
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] < 20.0
        is_neutral = not is_trending and not is_ranging
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > 1.5 * vol_sma[i] if vol_sma[i] > 1e-10 else False
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_bull = close[i] > donchian_upper[i-1] and close[i-1] <= donchian_upper[i-1]
        donchian_breakout_bear = close[i] < donchian_lower[i-1] and close[i-1] >= donchian_lower[i-1]
        
        # === DONCHIAN MEAN REVERSION SIGNALS (in ranging regime) ===
        donchian_range = donchian_upper[i] - donchian_lower[i]
        if donchian_range > 1e-10:
            price_position = (close[i] - donchian_lower[i]) / donchian_range
            near_lower = price_position < 0.15
            near_upper = price_position > 0.85
        else:
            near_lower = False
            near_upper = False
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 25.0
        rsi_ok_short = rsi[i] < 75.0
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === 4h KAMA TREND ===
        kama_bull = close[i] > kama_4h[i]
        kama_bear = close[i] < kama_4h[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic with ADX) ===
        desired_signal = 0.0
        signal_strength = 1.0
        
        if is_trending:
            # TREND REGIME (ADX > 25): Follow Donchian breakouts with HTF bias
            # LONG: breakout + volume + HTF bull + RSI ok + KAMA bull
            if donchian_breakout_bull and htf_bull and rsi_ok_long and kama_bull:
                if vol_confirmed:
                    desired_signal = SIZE
                    signal_strength = 1.0
                else:
                    desired_signal = SIZE * 0.7
                    signal_strength = 0.7
            # SHORT: breakout + volume + HTF bear + RSI ok + KAMA bear
            elif donchian_breakout_bear and htf_bear and rsi_ok_short and kama_bear:
                if vol_confirmed:
                    desired_signal = -SIZE
                    signal_strength = 1.0
                else:
                    desired_signal = -SIZE * 0.7
                    signal_strength = 0.7
            # Fallback: strong breakout overrides HTF
            elif donchian_breakout_bull and kama_bull and rsi[i] > 35.0:
                desired_signal = SIZE * 0.7
                signal_strength = 0.7
            elif donchian_breakout_bear and kama_bear and rsi[i] < 65.0:
                desired_signal = -SIZE * 0.7
                signal_strength = 0.7
                
        elif is_ranging:
            # RANGE REGIME (ADX < 20): Mean revert at Donchian bounds
            # LONG: near lower + RSI oversold + HTF not strongly bear
            if near_lower and rsi_oversold and not htf_bear:
                desired_signal = SIZE
                signal_strength = 1.0
            # SHORT: near upper + RSI overbought + HTF not strongly bull
            elif near_upper and rsi_overbought and not htf_bull:
                desired_signal = -SIZE
                signal_strength = 1.0
            # Fallback: extreme RSI mean reversion
            elif rsi[i] < 30.0 and kama_bull:
                desired_signal = SIZE * 0.7
                signal_strength = 0.7
            elif rsi[i] > 70.0 and kama_bear:
                desired_signal = -SIZE * 0.7
                signal_strength = 0.7
                
        else:
            # NEUTRAL REGIME (ADX 20-25): Reduced position size, wait for confirmation
            if donchian_breakout_bull and htf_bull and kama_bull:
                desired_signal = SIZE * 0.5
                signal_strength = 0.5
            elif donchian_breakout_bear and htf_bear and kama_bear:
                desired_signal = -SIZE * 0.5
                signal_strength = 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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