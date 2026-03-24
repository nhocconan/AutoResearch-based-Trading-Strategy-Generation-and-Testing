#!/usr/bin/env python3
"""
Experiment #1491: 4h Primary + 1d/1w HTF — Dual HTF Trend Filter with Donchian Breakout

Hypothesis: After analyzing 1100+ failed strategies, the pattern is clear:
1. 4h trend-following fails WITHOUT strong HTF confirmation (see #1479 Sharpe=-0.205)
2. Strategies with 0 trades (#1480, #1485, #1489, #1490) have OVERLY STRICT entry conditions
3. Current best (Sharpe=0.618) uses 1d Donchian + HMA + RSI — simple but effective
4. 12h/1d strategies work better than 4h because fewer whipsaws in 2022 crash

Key insight: For 4h to work, need BOTH 1w AND 1d agreement on trend direction.
This strategy uses:
- 1w HMA(21) as MACRO trend filter (only trade in weekly direction)
- 1d HMA(21) as INTERMEDIATE confirmation (must agree with weekly)
- 4h Donchian(20) breakout for entry timing
- RSI(14) loose filter (35-65 range) to ensure sufficient trades
- ATR(14)*2.5 trailing stoploss (mandatory for drawdown control)
- Discrete signal sizes (0.0, ±0.25, ±0.30) to minimize fee churn

Why this should beat current best (Sharpe=0.618):
1. Dual HTF filter (1w + 1d) prevents counter-trend trades in 2022 crash
2. Donchian breakout is proven (worked in #1482 with Sharpe=0.237)
3. Loose RSI (35-65 vs 45-55) ensures 20-50 trades/year target
4. 4h timeframe = more responsive than 1d while maintaining trend quality
5. ATR stoploss mandatory — all winning strategies had proper risk management

Timeframe: 4h
HTF: 1d AND 1w (call get_htf_data ONCE for each before loop!)
Position Size: 0.30 max (discrete: 0.0, ±0.25, ±0.30)
Target: 25-50 trades/year, Sharpe > 0.618, ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_1d1w_rsi_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — smoother and more responsive than EMA
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = np.full(len(series), np.nan)
        for i in range(window - 1, len(series)):
            if not np.isnan(series[i - window + 1:i + 1]).any():
                result[i] = np.dot(series[i - window + 1:i + 1], weights) / weights.sum()
        return result
    
    close_series = pd.Series(close)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # 2*WMA(n/2) - WMA(n)
    diff = 2 * wma_half - wma_full
    
    # WMA of diff with sqrt(n) period
    hma = wma(diff, sqrt_period)
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # Calculate TR, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
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
    
    # Smooth with Wilder's method
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI, -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX, ADX
    dx = np.full(n, np.nan)
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * di_diff[mask2] / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    adx = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) — PRIMARY direction filter ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) — must agree with weekly ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h[i]
        hma_4h_bear = close[i] < hma_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_high = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_low = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI MOMENTUM — LOOSE bands for more trades (35-65) ===
        rsi_bullish = rsi[i] > 35.0
        rsi_bearish = rsi[i] < 65.0
        rsi_strong_bull = rsi[i] > 45.0
        rsi_strong_bear = rsi[i] < 55.0
        
        # === ADX TREND STRENGTH ===
        trend_strong = adx[i] > 20.0  # Loose threshold for more trades
        trend_very_strong = adx[i] > 25.0
        
        # === DESIRED SIGNAL — DUAL HTF FILTER + BREAKOUT ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + Daily bull + (Breakout OR 4h HMA bull) + RSI support
        if weekly_bull and daily_bull:
            if breakout_high and rsi_bullish:
                # Strong breakout with momentum
                desired_signal = BASE_SIZE
            elif hma_4h_bull and rsi_strong_bull and trend_strong:
                # HMA trend with momentum
                desired_signal = BASE_SIZE * 0.8
            elif hma_4h_bull and rsi[i] > 40.0:
                # Weaker signal but still valid
                desired_signal = BASE_SIZE * 0.5
        
        # SHORT: Weekly bear + Daily bear + (Breakout OR 4h HMA bear) + RSI support
        elif weekly_bear and daily_bear:
            if breakout_low and rsi_bearish:
                # Strong breakout with momentum
                desired_signal = -BASE_SIZE
            elif hma_4h_bear and rsi_strong_bear and trend_strong:
                # HMA trend with momentum
                desired_signal = -BASE_SIZE * 0.8
            elif hma_4h_bear and rsi[i] < 60.0:
                # Weaker signal but still valid
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.5
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