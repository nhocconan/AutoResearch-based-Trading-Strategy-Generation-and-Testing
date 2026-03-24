#!/usr/bin/env python3
"""
Experiment #375: 6h Primary + 12h/1d HTF — Z-Score Mean Reversion with Regime Filter

Hypothesis: 6h timeframe sits in a sweet spot between noise (lower TF) and lag (higher TF).
Previous 6h strategies failed due to overly complex regime detection. This version uses:
1. SIMPLE regime: 1d HMA slope direction (bull/bear)
2. Z-score(20) mean reversion with RSI confirmation
3. Choppiness Index to avoid trending whipsaws
4. Asymmetric entries: long bias in bull, short bias in bear
5. Volume spike confirmation for breakouts

Key differences from failed #367, #371:
- Simpler regime (HMA slope vs ADX+CHOP combo)
- Z-score extremes (proven mean reversion signal)
- Fewer confluence requirements (max 3 filters)
- Proper HTF alignment with 12h + 1d

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test
Position sizing: 0.25 base, 0.30 with HTF alignment
Stoploss: 2.5x ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_zscore_regime_hma_rsi_12h1d_v1"
timeframe = "6h"
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

def calculate_zscore(close, period=20):
    """Z-Score for mean reversion detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    rolling_mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    rolling_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    zscore[:] = np.nan
    for i in range(period-1, n):
        if rolling_std[i] > 1e-10:
            zscore[i] = (close[i] - rolling_mean[i]) / rolling_std[i]
    
    return zscore

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope direction"""
    n = len(hma)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i-lookback]):
            slope[i] = hma[i] - hma[i-lookback]
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate HMA slopes for HTF trend direction
    hma_1d_slope_raw = calculate_hma_slope(hma_1d_raw, lookback=3)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    hma_6h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    zscore = calculate_zscore(close, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_volume_sma(volume, 20)
    
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
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (SIMPLE: HTF HMA slope) ===
        htf_1d_bull = hma_1d_slope_aligned[i] > 0 if not np.isnan(hma_1d_slope_aligned[i]) else close[i] > hma_1d_aligned[i]
        htf_1d_bear = hma_1d_slope_aligned[i] < 0 if not np.isnan(hma_1d_slope_aligned[i]) else close[i] < hma_1d_aligned[i]
        
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = not np.isnan(chop[i]) and chop[i] > 55.0
        is_trending = not np.isnan(chop[i]) and chop[i] < 45.0
        
        # === Z-SCORE EXTREMES (mean reversion signals) ===
        zscore_extreme_low = zscore[i] < -1.5
        zscore_extreme_high = zscore[i] > 1.5
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === HMA CROSSOVER ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_6h_fast[i]) and not np.isnan(hma_6h_fast[i-1]):
            if not np.isnan(hma_6h[i]) and not np.isnan(hma_6h[i-1]):
                if hma_6h_fast[i-1] <= hma_6h[i-1] and hma_6h_fast[i] > hma_6h[i]:
                    hma_cross_long = True
                if hma_6h_fast[i-1] >= hma_6h[i-1] and hma_6h_fast[i] < hma_6h[i]:
                    hma_cross_short = True
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = False
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES (asymmetric: prefer longs in bull regime)
        if htf_1d_bull or htf_12h_bull:
            # Mean reversion long: Z-score extreme + RSI oversold + above SMA200
            if zscore_extreme_low and rsi_oversold and above_sma200:
                if is_choppy:
                    desired_signal = SIZE_STRONG if vol_confirm else SIZE_BASE
                else:
                    desired_signal = SIZE_BASE
            
            # HMA cross long with HTF confirmation
            elif hma_cross_long and htf_12h_bull:
                desired_signal = SIZE_BASE
        
        # SHORT ENTRIES (asymmetric: prefer shorts in bear regime)
        elif htf_1d_bear or htf_12h_bear:
            # Mean reversion short: Z-score extreme + RSI overbought + below SMA200
            if zscore_extreme_high and rsi_overbought and below_sma200:
                if is_choppy:
                    desired_signal = -SIZE_STRONG if vol_confirm else -SIZE_BASE
                else:
                    desired_signal = -SIZE_BASE
            
            # HMA cross short with HTF confirmation
            elif hma_cross_short and htf_12h_bear:
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