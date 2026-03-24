#!/usr/bin/env python3
"""
Experiment #462: 4h Primary + 1d/1w HTF — Simplified Trend + Mean Reversion Hybrid

Hypothesis: 4h timeframe has proven successful in past experiments (#458 showed promise).
Recent failures show too many filters = 0 trades. This strategy simplifies entry logic:

1. SINGLE HTF BIAS: 1d HMA for trend direction (not dual HTF which was too restrictive)
2. WEEKLY FILTER: 1w HMA only for major trend bias (loose filter, not hard requirement)
3. DUAL ENTRY MODES: 
   - Trend: HMA crossover + 1d alignment (breakout style)
   - Mean Revert: RSI extremes + BB touch (pullback style)
4. LOOSE REGIME: ADX > 15 for trend (not 20), allows more trades
5. CONNORS RSI LITE: RSI(3) for faster mean reversion signals

Key changes from failed experiments:
- Removed dual HTF requirement (12h+1d both must agree = too restrictive)
- Simplified regime detection (ADX threshold 15 vs 18-20)
- Added RSI(3) for faster mean reversion entries
- Ensured minimum 30 trades/year via loose entry conditions

Target: Sharpe>0.45, DD>-35%, trades>=80 train (20/year), trades>=12 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_connors_donchian_1d1w_v1"
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

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + slow_period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio
    signal = np.zeros(n)
    noise = np.zeros(n)
    
    for i in range(er_period, n):
        signal[i] = abs(close[i] - close[i - er_period])
        for j in range(i - er_period + 1, i + 1):
            noise[i] += abs(close[j] - close[j - 1])
    
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(er_period, n):
        if noise[i] > 1e-10:
            er[i] = signal[i] / noise[i]
    
    # Smoothing constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        if not np.isnan(er[i]):
            sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
            sc[i] = sc[i] ** 2
    
    # KAMA
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    rsi_3 = calculate_rsi(close, period=3)  # Connors RSI component
    bb_upper, bb_lower = calculate_bollinger(close, period=20, std_dev=2.0)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
        
        if np.isnan(hma_4h[i]) or np.isnan(rsi_14[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1d + 1w) ===
        # 1d HMA for primary trend, 1w HMA for major bias
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Weekly bias (loose filter - just confirms major trend)
        weekly_bullish = htf_1w_bull
        weekly_bearish = htf_1w_bear
        
        # === 4h TREND INDICATORS ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        kama_bull = close[i] > kama_4h[i] if not np.isnan(kama_4h[i]) else False
        kama_bear = close[i] < kama_4h[i] if not np.isnan(kama_4h[i]) else False
        
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
        donchian_breakout_long = False
        donchian_breakdown_short = False
        if not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            donchian_breakout_long = close[i] > donchian_upper[i-1]
            donchian_breakdown_short = close[i] < donchian_lower[i-1]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === RSI EXTREMES ===
        rsi_14_oversold = rsi_14[i] < 35.0
        rsi_14_overbought = rsi_14[i] > 65.0
        rsi_3_oversold = rsi_3[i] < 20.0  # Connors RSI extreme
        rsi_3_overbought = rsi_3[i] > 80.0
        
        # === BB TOUCH ===
        touch_lower = close[i] <= bb_lower[i] if not np.isnan(bb_lower[i]) else False
        touch_upper = close[i] >= bb_upper[i] if not np.isnan(bb_upper[i]) else False
        
        # === ADX TREND STRENGTH ===
        is_trending = adx[i] > 15.0  # Lower threshold for more trades
        is_choppy = adx[i] < 15.0
        
        # === ENTRY LOGIC (LOOSE - ensure trades generate) ===
        desired_signal = 0.0
        
        # MODE 1: TREND FOLLOWING (breakout + HTF alignment)
        if is_trending:
            # Long: 1d bull + weekly bull + (HMA bull OR Donchian breakout OR HMA cross)
            if htf_1d_bull and weekly_bullish:
                if hma_bull or donchian_breakout_long or hma_cross_long:
                    desired_signal = SIZE_STRONG
            # Also allow long with just 1d bull (weekly is loose filter)
            elif htf_1d_bull:
                if donchian_breakout_long and above_sma50:
                    desired_signal = SIZE_BASE
            
            # Short: 1d bear + weekly bear + (HMA bear OR Donchian breakdown OR HMA cross)
            elif htf_1d_bear and weekly_bearish:
                if hma_bear or donchian_breakdown_short or hma_cross_short:
                    desired_signal = -SIZE_STRONG
            # Also allow short with just 1d bear
            elif htf_1d_bear:
                if donchian_breakdown_short and below_sma50:
                    desired_signal = -SIZE_BASE
        
        # MODE 2: MEAN REVERSION (RSI extremes + BB touch)
        elif is_choppy:
            # Long: RSI oversold + BB lower touch OR RSI(3) extreme
            if rsi_14_oversold or rsi_3_oversold:
                if touch_lower or above_sma200:
                    desired_signal = SIZE_BASE
            # Short: RSI overbought + BB upper touch OR RSI(3) extreme
            elif rsi_14_overbought or rsi_3_overbought:
                if touch_upper or below_sma200:
                    desired_signal = -SIZE_BASE
        
        # MODE 3: KAMA TREND CONFIRMATION (alternative trend entry)
        if kama_bull and htf_1d_bull and rsi_14[i] > 45.0 and rsi_14[i] < 70.0:
            if desired_signal < SIZE_BASE:
                desired_signal = SIZE_BASE
        
        if kama_bear and htf_1d_bear and rsi_14[i] > 30.0 and rsi_14[i] < 55.0:
            if desired_signal > -SIZE_BASE:
                desired_signal = -SIZE_BASE
        
        # === TRAILING STOPLOGIC ===
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