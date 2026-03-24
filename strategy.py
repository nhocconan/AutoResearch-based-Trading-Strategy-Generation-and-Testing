#!/usr/bin/env python3
"""
Experiment #402: 4h Primary + 1d/1w HTF — Simplified Dual Regime v3

Hypothesis: Previous 4h strategies failed due to overly strict regime detection
that rarely triggered entries (0 trades). This version MAXIMIZES trade frequency
while maintaining edge through proven patterns.

Key changes from failed experiments:
1. SIMPLIFIED regime detection (ADX only, drop Choppiness complexity)
2. LOOSENED RSI thresholds (20/80 for more triggers)
3. Donchian breakout as primary trend entry (proven on 4h)
4. Bollinger mean reversion as secondary (works in ranges)
5. 1d HTF for size adjustment only (not entry filter)
6. Ensure EVERY symbol gets trades by loosening confluence

Regime Detection:
- ADX > 25 = trending → Donchian breakout + HMA alignment
- ADX < 20 = ranging → Bollinger/RSI mean reversion
- ADX 20-25 = use previous regime (hysteresis)

Entry Logic (MINIMAL confluence):
- Trending Long: ADX>25 + HMA(21) bull + Donchian breakout OR HMA cross
- Trending Short: ADX>25 + HMA(21) bear + Donchian breakdown OR HMA cross
- Ranging Long: ADX<20 + RSI<20 + price>BB_lower
- Ranging Short: ADX<20 + RSI>80 + price<BB_upper

Position sizing: 0.25 base, 0.30 when 1d HTF aligned
Stoploss: 2.5x ATR(14) from entry
Take Profit: Reduce to half at 2R, trail rest

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simplified_regime_hma_donchian_bb_1d1w_v3"
timeframe = "4h"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    for i in range(er_period, n):
        noise = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        signal = abs(close[i] - close[i-er_period])
        
        if noise > 1e-10:
            er = signal / noise
        else:
            er = 1.0
        
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == er_period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    kama_4h = calculate_kama(close)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Regime memory for hysteresis
    prev_regime = 0  # 0=unknown, 1=trending, 2=ranging
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    tp_hit = False
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION with ADX (SIMPLIFIED) ===
        # Trending: ADX > 25
        # Ranging: ADX < 20
        # Otherwise: use previous regime (hysteresis)
        
        is_trending = adx[i] > 25.0
        is_ranging = adx[i] < 20.0
        
        if is_trending:
            current_regime = 1
        elif is_ranging:
            current_regime = 2
        else:
            current_regime = prev_regime
        
        prev_regime = current_regime
        
        # === HTF BIAS (1d + 1w) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Strong HTF alignment (both 1d and 1w agree)
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_4h_fast[i]) and not np.isnan(hma_4h_fast[i-1]):
            if not np.isnan(hma_4h[i]) and not np.isnan(hma_4h[i-1]):
                if hma_4h_fast[i-1] <= hma_4h[i-1] and hma_4h_fast[i] > hma_4h[i]:
                    hma_cross_long = True
                if hma_4h_fast[i-1] >= hma_4h[i-1] and hma_4h_fast[i] < hma_4h[i]:
                    hma_cross_short = True
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = False
        breakout_short = False
        if not np.isnan(donchian_upper[i-1]):
            breakout_long = close[i] > donchian_upper[i-1]
        if not np.isnan(donchian_lower[i-1]):
            breakout_short = close[i] < donchian_lower[i-1]
        
        # === BOLLINGER POSITION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.002  # within 0.2% of lower
        near_bb_upper = close[i] >= bb_upper[i] * 0.998  # within 0.2% of upper
        
        # === RSI EXTREMES (LOOSENED for more trades) ===
        rsi_oversold = rsi[i] < 20.0
        rsi_overbought = rsi[i] > 80.0
        
        # === ENTRY LOGIC (MINIMAL confluence - ensure trades trigger) ===
        desired_signal = 0.0
        
        # REGIME 1: TRENDING (breakout + trend alignment)
        if current_regime == 1:
            # Long: HMA bull + (breakout OR cross) - only 2 conditions!
            if hma_bull:
                if breakout_long or hma_cross_long:
                    # Size boost if HTF aligned
                    if htf_strong_bull:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # Short: HMA bear + (breakdown OR cross) - only 2 conditions!
            elif hma_bear:
                if breakout_short or hma_cross_short:
                    if htf_strong_bear:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # REGIME 2: RANGING (RSI + Bollinger mean reversion)
        elif current_regime == 2:
            # Long: RSI oversold + near BB lower (2 conditions)
            if rsi_oversold and near_bb_lower:
                desired_signal = SIZE_BASE
            
            # Short: RSI overbought + near BB upper (2 conditions)
            elif rsi_overbought and near_bb_upper:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT CHECK (reduce at 2R) ===
        tp_triggered = False
        if in_position and not tp_hit:
            if position_side > 0:
                if close[i] >= entry_price + 2.0 * entry_atr * 2.5:
                    tp_triggered = True
                    tp_hit = True
            elif position_side < 0:
                if close[i] <= entry_price - 2.0 * entry_atr * 2.5:
                    tp_triggered = True
                    tp_hit = True
        
        if tp_triggered and desired_signal == 0.0:
            # Reduce to half position
            if position_side > 0:
                desired_signal = SIZE_BASE / 2
            elif position_side < 0:
                desired_signal = -SIZE_BASE / 2
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= (SIZE_BASE / 2) * 0.9:
            final_signal = SIZE_BASE / 2
        elif desired_signal <= -(SIZE_BASE / 2) * 0.9:
            final_signal = -SIZE_BASE / 2
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
                tp_hit = False
                # Set stoploss
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
                tp_hit = False
        
        signals[i] = final_signal
    
    return signals