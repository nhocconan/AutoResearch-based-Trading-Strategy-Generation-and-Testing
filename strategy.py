#!/usr/bin/env python3
"""
Experiment #1327: 1d Primary + 1w HTF — Regime-Adaptive Mean Reversion + Trend

Hypothesis: Daily timeframe with weekly trend filter can capture both:
1. Mean reversion in ranging markets (RSI extremes + BB bands)
2. Trend continuation in directional markets (HMA slope + pullback)

Key innovations vs failed experiments:
1. Use 1w HMA for macro regime (bull/bear) — slower than 1d, more stable
2. Adaptive RSI bands: wider in trends (25-75), tighter in ranges (30-70)
3. Volatility expansion filter: ATR(7)/ATR(21) > 1.5 = breakout potential
4. BB %B for entry timing: enter when %B < 0.1 (oversold) or > 0.9 (overbought)
5. Ensure trade frequency: loose enough RSI bands to hit 20-50 trades/year

Timeframe: 1d
HTF: 1w (for macro trend direction)
Size: 0.25-0.30 discrete
Target: Sharpe > 0.612, trades >= 40 train, >= 5 test per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_rsi_bb_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
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
    rsi[~mask] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with %B indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # %B = (close - lower) / (upper - lower)
    pct_b = np.full(n, np.nan)
    mask = (upper - lower) > 1e-10
    pct_b[mask] = (close[mask] - lower[mask]) / (upper[mask] - lower[mask])
    
    return upper, lower, pct_b

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

def calculate_sma(close, period=200):
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1w HMA slope (trend direction)
    hma_1w_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]):
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-1]) / hma_1w_aligned[i-1]
    
    # Calculate primary (1d) indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_21 = calculate_atr(high, low, close, period=21)
    sma_200 = calculate_sma(close, period=200)
    bb_upper, bb_lower, pct_b = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # ATR ratio for volatility expansion
    atr_ratio = np.full(n, np.nan)
    mask = atr_21 > 1e-10
    atr_ratio[mask] = atr_7[mask] / atr_21[mask]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(sma_200[i]) or np.isnan(pct_b[i]):
            signals[i] = 0.0
            continue
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO REGIME (1w HMA slope) ===
        # Positive slope = bull regime, Negative slope = bear regime
        bull_regime = hma_1w_slope[i] > 0.0
        bear_regime = hma_1w_slope[i] < 0.0
        
        # === LOCAL TREND (1d HMA) ===
        hma_bull = hma_21[i] > hma_50[i]
        hma_bear = hma_21[i] < hma_50[i]
        
        # === VOLATILITY REGIME ===
        vol_expansion = atr_ratio[i] > 1.5  # ATR(7) > 1.5 * ATR(21)
        vol_compression = atr_ratio[i] < 0.8
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY CONDITIONS
        if bull_regime or hma_bull:
            # Mean reversion: RSI oversold + BB %B low
            if rsi[i] < 35.0 and pct_b[i] < 0.15:
                desired_signal = BASE_SIZE
            # Pullback in uptrend: RSI 35-50 + above SMA200
            elif 35.0 <= rsi[i] <= 50.0 and above_sma200 and hma_bull:
                desired_signal = BASE_SIZE
            # Volatility expansion breakout: RSI > 50 + vol expansion
            elif rsi[i] > 50.0 and rsi[i] < 65.0 and vol_expansion and above_sma200:
                desired_signal = BASE_SIZE
            # Deep oversold bounce (even in bear regime)
            elif rsi[i] < 25.0 and pct_b[i] < 0.05:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRY CONDITIONS
        elif bear_regime or hma_bear:
            # Mean reversion: RSI overbought + BB %B high
            if rsi[i] > 65.0 and pct_b[i] > 0.85:
                desired_signal = -BASE_SIZE
            # Pullback in downtrend: RSI 50-65 + below SMA200
            elif 50.0 <= rsi[i] <= 65.0 and below_sma200 and hma_bear:
                desired_signal = -BASE_SIZE
            # Volatility expansion breakdown: RSI < 50 + vol expansion
            elif rsi[i] < 50.0 and rsi[i] > 35.0 and vol_expansion and below_sma200:
                desired_signal = -BASE_SIZE
            # Deep overbought rejection (even in bull regime)
            elif rsi[i] > 75.0 and pct_b[i] > 0.95:
                desired_signal = -BASE_SIZE
        
        # === RANGE MARKET: Pure mean reversion ===
        if not bull_regime and not bear_regime:
            # Long at extreme oversold
            if rsi[i] < 30.0 and pct_b[i] < 0.1:
                desired_signal = BASE_SIZE
            # Short at extreme overbought
            elif rsi[i] > 70.0 and pct_b[i] > 0.9:
                desired_signal = -BASE_SIZE
        
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
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
            final_signal = -BASE_SIZE
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