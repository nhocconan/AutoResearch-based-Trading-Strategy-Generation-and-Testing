#!/usr/bin/env python3
"""
Experiment #363: 6h Primary + 1d/1w HTF — Triple HMA Trend Alignment v1

Hypothesis: Previous 6h strategies failed because they tried mean-reversion on a 
timeframe that trends. 6h is long enough for multi-day trends to persist. This 
strategy uses TRIPLE HMA alignment (1w > 1d > 6h) for strong trend confirmation,
with entries on pullbacks to 6h HMA(21).

Key insights from failures:
- #360 (mean reversion): Sharpe=-12.1 — mean reversion dies on 6h
- #351, #355, #362: All negative Sharpe — complex conditions or wrong regime
- #352 (12h ADX/Chop): Sharpe=-0.186 but +24% return — regime detection has merit

New approach:
1. Weekly HMA(21) slope = primary bias (ONLY trade in weekly trend direction)
2. Daily HMA(21) = secondary confirmation (must align with weekly)
3. 6h HMA(21) + HMA(10) cross = entry trigger on pullback
4. ADX(14) > 18 = trend strength filter (looser than 25 to get more trades)
5. ATR(14) trailing stop = 2.5x from entry

Why this should work:
- Weekly filter prevents counter-trend trades (major failure mode)
- Triple alignment = high conviction, fewer but better trades
- Pullback entries = better risk/reward than breakouts
- ADX > 18 (not 25) = more trades while still filtering chop

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_triple_hma_trend_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA, less lag"""
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=21)
    hma_6h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(hma_6h_fast[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY TREND BIAS (primary filter) ===
        # Weekly HMA slope: compare current to 3 bars ago
        weekly_bull = False
        weekly_bear = False
        if i >= 3 and not np.isnan(hma_1w_aligned[i-3]):
            if hma_1w_aligned[i] > hma_1w_aligned[i-3] * 1.002:  # 0.2% threshold
                weekly_bull = True
            elif hma_1w_aligned[i] < hma_1w_aligned[i-3] * 0.998:
                weekly_bear = True
        
        # Price relative to weekly HMA
        price_above_1w = close[i] > hma_1w_aligned[i] * 1.001
        price_below_1w = close[i] < hma_1w_aligned[i] * 0.999
        
        # === DAILY TREND CONFIRMATION (secondary filter) ===
        daily_bull = close[i] > hma_1d_aligned[i] * 1.001
        daily_bear = close[i] < hma_1d_aligned[i] * 0.999
        
        # Daily HMA slope
        daily_slope_bull = False
        daily_slope_bear = False
        if i >= 3 and not np.isnan(hma_1d_aligned[i-3]):
            if hma_1d_aligned[i] > hma_1d_aligned[i-3] * 1.001:
                daily_slope_bull = True
            elif hma_1d_aligned[i] < hma_1d_aligned[i-3] * 0.999:
                daily_slope_bear = True
        
        # === 6h HMA CROSSOVER (entry trigger) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_6h_fast[i-1]) and not np.isnan(hma_6h[i-1]):
            if hma_6h_fast[i-1] <= hma_6h[i-1] and hma_6h_fast[i] > hma_6h[i]:
                hma_cross_long = True
            if hma_6h_fast[i-1] >= hma_6h[i-1] and hma_6h_fast[i] < hma_6h[i]:
                hma_cross_short = True
        
        # === PULLBACK ENTRY (price near 6h HMA) ===
        # Long: price pulled back to HMA but weekly/daily still bull
        pullback_long = False
        pullback_short = False
        
        hma_distance = (close[i] - hma_6h[i]) / hma_6h[i] if hma_6h[i] > 0 else 0
        
        # Within 1.5% of HMA = pullback zone
        if abs(hma_distance) < 0.015:
            if weekly_bull and daily_bull and hma_distance < 0:
                pullback_long = True
            if weekly_bear and daily_bear and hma_distance > 0:
                pullback_short = True
        
        # === ADX TREND STRENGTH ===
        trend_strong = adx[i] > 18.0  # Looser threshold for more trades
        
        # === RSI FILTER (avoid extreme overbought/oversold entries) ===
        rsi_ok_long = rsi[i] < 70.0  # Don't buy at extreme
        rsi_ok_short = rsi[i] > 30.0  # Don't sell at extreme
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Weekly bull + Daily bull + (HMA cross OR pullback) + ADX + RSI ok
        if weekly_bull and daily_bull and trend_strong and rsi_ok_long:
            if hma_cross_long:
                desired_signal = SIZE_STRONG
            elif pullback_long:
                desired_signal = SIZE_BASE
        
        # SHORT: Weekly bear + Daily bear + (HMA cross OR pullback) + ADX + RSI ok
        elif weekly_bear and daily_bear and trend_strong and rsi_ok_short:
            if hma_cross_short:
                desired_signal = -SIZE_STRONG
            elif pullback_short:
                desired_signal = -SIZE_BASE
        
        # === TRAILING STOPLOSS CHECK (2.5x ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest price for trailing
            if close[i] > highest_price:
                highest_price = close[i]
                stop_price = highest_price - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Update lowest price for trailing
            if close[i] < lowest_price:
                lowest_price = close[i]
                stop_price = lowest_price + 2.5 * entry_atr
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_price = close[i]
                lowest_price = close[i]
                # Set initial stoploss
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
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = final_signal
    
    return signals