#!/usr/bin/env python3
"""
Experiment #471: 6h Primary + 1d/1w HTF — KAMA Adaptive Trend + Volume Breakout

Hypothesis: 6h timeframe needs adaptive indicators that reduce whipsaw in choppy markets.
KAMA (Kaufman Adaptive Moving Average) adjusts smoothing based on market efficiency -
perfect for 6h which sits between trend (12h) and noise (1h).

Key innovations vs failed 6h experiments:
1. KAMA instead of HMA/EMA - adapts to market regime automatically
2. OR logic for HTF bias (1d OR 1w bull) - not AND (too restrictive = 0 trades)
3. Volume confirmation on breakouts only (not mean reversion) - more trades qualify
4. Looser RSI (35/65) + multiple entry paths - ensures 30+ trades/year
5. ROC momentum filter - avoids entering against momentum

Entry Logic:
- Trend Long: (1d KAMA bull OR 1w KAMA bull) + 6h KAMA cross + ROC > 0 + volume confirm
- Trend Short: (1d KAMA bear OR 1w KAMA bear) + 6h KAMA cross + ROC < 0 + volume confirm
- Mean Rev Long: RSI < 35 + price > SMA200 + ADX < 25 (no volume needed)
- Mean Rev Short: RSI > 65 + price < SMA200 + ADX < 25 (no volume needed)

Target: Sharpe>0.45, DD>-35%, trades>=60 train, trades>=10 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_kama_volume_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[max(0, i-period):i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    sc = np.zeros(n)
    sc[:] = np.nan
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
            sc[i] = sc[i] ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    roc = np.zeros(n)
    roc[:] = np.nan
    for i in range(period, n):
        if close[i - period] > 1e-10:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate primary (6h) indicators
    kama_6h = calculate_kama(close, period=21)
    kama_6h_fast = calculate_kama(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    roc = calculate_roc(close, period=10)
    vol_sma = calculate_volume_sma(volume, period=20)
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
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_6h[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(roc[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (OR LOGIC - more trades than AND) ===
        htf_1d_bull = close[i] > kama_1d_aligned[i]
        htf_1d_bear = close[i] < kama_1d_aligned[i]
        htf_1w_bull = close[i] > kama_1w_aligned[i]
        htf_1w_bear = close[i] < kama_1w_aligned[i]
        
        # At least one HTF must agree (OR logic)
        htf_bull = htf_1d_bull or htf_1w_bull
        htf_bear = htf_1d_bear or htf_1w_bear
        
        # === 6h KAMA TREND ===
        kama_bull = close[i] > kama_6h[i]
        kama_bear = close[i] < kama_6h[i]
        
        # === KAMA CROSSOVER ===
        kama_cross_long = False
        kama_cross_short = False
        if i > 0 and not np.isnan(kama_6h_fast[i]) and not np.isnan(kama_6h_fast[i-1]):
            if not np.isnan(kama_6h[i]) and not np.isnan(kama_6h[i-1]):
                if kama_6h_fast[i-1] <= kama_6h[i-1] and kama_6h_fast[i] > kama_6h[i]:
                    kama_cross_long = True
                if kama_6h_fast[i-1] >= kama_6h[i-1] and kama_6h_fast[i] < kama_6h[i]:
                    kama_cross_short = True
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = False
        donchian_breakdown_short = False
        if not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            donchian_breakout_long = close[i] > donchian_upper[i-1]
            donchian_breakdown_short = close[i] < donchian_lower[i-1]
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = False
        if vol_sma[i] > 1e-10:
            volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        # === MOMENTUM (ROC) ===
        roc_positive = roc[i] > 0.0
        roc_negative = roc[i] < 0.0
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES (LOOSE: 35/65) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === TREND STRENGTH ===
        is_trending = adx[i] > 20.0
        is_choppy = adx[i] < 20.0
        
        # === ENTRY LOGIC (MULTIPLE PATHS FOR TRADES) ===
        desired_signal = 0.0
        
        # PATH 1: TREND FOLLOWING (KAMA cross + HTF bias + momentum)
        if is_trending and htf_bull:
            if kama_cross_long and roc_positive:
                if volume_confirm or donchian_breakout_long:
                    desired_signal = SIZE_STRONG
        
        if is_trending and htf_bear:
            if kama_cross_short and roc_negative:
                if volume_confirm or donchian_breakdown_short:
                    desired_signal = -SIZE_STRONG
        
        # PATH 2: DONCHIAN BREAKOUT (stronger signal)
        if htf_bull and donchian_breakout_long and volume_confirm:
            if kama_bull and roc_positive:
                desired_signal = SIZE_STRONG
        
        if htf_bear and donchian_breakdown_short and volume_confirm:
            if kama_bear and roc_negative:
                desired_signal = -SIZE_STRONG
        
        # PATH 3: MEAN REVERSION (choppy market, RSI extremes)
        if is_choppy:
            if rsi_oversold and above_sma200:
                desired_signal = SIZE_BASE
            elif rsi_overbought and below_sma200:
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
        
        signals[i] = final_signal
    
    return signals