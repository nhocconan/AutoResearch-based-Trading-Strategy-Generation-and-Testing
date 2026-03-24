#!/usr/bin/env python3
"""
Experiment #488: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + ADX Filter

Hypothesis: Previous 4h strategies failed due to HMA whipsaw in volatile conditions.
KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - slows in chop,
speeds up in trends. Combined with ADX filter and moderate RSI pullbacks, this should:
1. Reduce false signals in ranging markets (KAMA + ADX + Choppiness)
2. Generate sufficient trades via RSI pullback entries (not just breakouts)
3. Survive 2022 crash with adaptive position sizing and wider stops

Key improvements from #478:
- KAMA instead of HMA (adapts to volatility, less whipsaw)
- ADX > 20 filter (only trade when trending)
- Choppiness Index < 61.8 (avoid ranging markets)
- RSI 40/60 pullback levels (more trades than 30/70 extremes)
- 2.5x ATR stoploss (wider room, fewer premature exits)
- Dual HTF: 12h for intermediate trend, 1d for macro bias

Target: Sharpe > 0.45, trades >= 100 train, trades >= 15 test, DD > -40%
Timeframe: 4h (proven to generate trades, lower fee drag than 15m/30m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_chop_rsi_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    Adapts to market efficiency - smooth in noise, fast in trends
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    for i in range(period, n):
        if i < period:
            continue
        
        # Price change over period
        price_change = abs(close[i] - close[i - period])
        
        # Sum of absolute price changes (volatility)
        if i < period:
            volatility = price_change
        else:
            volatility = sum(abs(close[j] - close[j-1]) for j in range(i - period + 1, i + 1))
        
        # Efficiency Ratio (0 = noise, 1 = trend)
        if volatility > 1e-10:
            er = price_change / volatility
        else:
            er = 0.0
        
        # Smoothing constants
        fast_sc = (2.0 / (fast + 1.0)) ** 2
        slow_sc = (2.0 / (slow + 1.0)) ** 2
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Initialize KAMA
        if i == period:
            kama[i] = close[i]
        elif i == period + 1:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 3:
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
        if down_move > up_move and down_move > 0:
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
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    > 61.8 = ranging (mean revert), < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest = max(high[i-period+1:i+1])
        lowest = min(low[i-period+1:i+1])
        
        if highest > lowest and (highest - lowest) > 1e-10:
            atr_sum = sum(
                max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                for j in range(i - period + 1, i + 1)
            )
            chop[i] = 100.0 * (atr_sum / (highest - lowest)) / (np.log10(period) if period > 1 else 1)
        else:
            chop[i] = 50.0
    
    return chop

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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
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
        
        if np.isnan(kama_4h[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
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
        
        if np.isnan(sma_50[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (12h + 1d agreement) ===
        htf_bull = close[i] > kama_12h_aligned[i] and close[i] > kama_1d_aligned[i]
        htf_bear = close[i] < kama_12h_aligned[i] and close[i] < kama_1d_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # === 4h KAMA TREND ===
        kama_bull = close[i] > kama_4h[i]
        kama_bear = close[i] < kama_4h[i]
        
        # === ADX TREND STRENGTH ===
        trending = adx[i] > 20.0  # ADX > 20 = trending market
        strong_trend = adx[i] > 25.0
        
        # === CHOPPINESS FILTER ===
        chopping = chop[i] > 61.8  # > 61.8 = ranging market
        not_chopping = chop[i] < 55.0  # < 55 = trending market
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI PULLBACK LEVELS (moderate, not extreme) ===
        rsi_pullback_long = 40.0 < rsi[i] < 55.0
        rsi_pullback_short = 45.0 < rsi[i] < 60.0
        rsi_momentum_long = rsi[i] > 55.0 and rsi[i-1] <= 55.0
        rsi_momentum_short = rsi[i] < 45.0 and rsi[i-1] >= 45.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND LONG: HTF bull + trending + KAMA bull + RSI pullback or momentum
        if htf_bull and not_chopping:
            if kama_bull and above_sma50:
                if rsi_momentum_long:
                    desired_signal = SIZE_STRONG if strong_trend else SIZE_BASE
                elif rsi_pullback_long and adx[i] > 18.0:
                    desired_signal = SIZE_BASE
        
        # TREND SHORT: HTF bear + trending + KAMA bear + RSI pullback or momentum
        elif htf_bear and not_chopping:
            if kama_bear and below_sma50:
                if rsi_momentum_short:
                    desired_signal = -SIZE_STRONG if strong_trend else -SIZE_BASE
                elif rsi_pullback_short and adx[i] > 18.0:
                    desired_signal = -SIZE_BASE
        
        # MEAN REVERSION in neutral HTF (only when not strongly trending)
        if desired_signal == 0.0 and htf_neutral:
            if chopping and rsi[i] < 35.0 and above_sma200:
                desired_signal = SIZE_BASE * 0.8
            elif chopping and rsi[i] > 65.0 and below_sma200:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update trailing high
            if close[i] > highest_price:
                highest_price = close[i]
            # Check stoploss
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Update trailing low
            if close[i] < lowest_price:
                lowest_price = close[i]
            # Check stoploss
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.8
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
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                    highest_price = entry_price
                else:
                    stop_price = entry_price + 2.5 * entry_atr
                    lowest_price = entry_price
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