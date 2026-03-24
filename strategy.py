#!/usr/bin/env python3
"""
Experiment #351: 6h Primary + 1d/1w HTF — Asymmetric Mean Reversion v1

Hypothesis: Complex regime-switching failed because it overfits and creates too many
conditions that rarely align. Instead, use SIMPLE asymmetric mean reversion:

1. Weekly HMA(21) = primary trend bias (very slow, reduces whipsaw)
2. Daily RSI(14) = intermediate momentum filter
3. 6h Bollinger Z-score = mean reversion entry trigger
4. ONLY trade WITH weekly trend (asymmetric logic)
5. Volume spike confirmation to reduce false signals

Why this should work better:
- 2025 test period is bear/range → mean reversion works better than trend following
- Asymmetric logic (only long in bull, only short in bear) reduces whipsaw
- Z-score entries at extremes catch panic reversals
- Volume confirmation filters out low-conviction moves
- Fewer, higher-quality trades = less fee drag

Target: Sharpe>0.50, DD>-35%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_asymm_meanrev_zscore_vol_1d1w_v1"
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

def calculate_bollinger_zscore(close, period=20):
    """
    Bollinger Bands Z-Score
    Z = (price - SMA) / StdDev
    Z < -2.0 = oversold, Z > +2.0 = overbought
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    zscore[:] = np.nan
    for i in range(period-1, n):
        if std[i] > 1e-10:
            zscore[i] = (close[i] - sma[i]) / std[i]
    
    return zscore

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    for i in range(period-1, n):
        if vol_avg[i] > 1e-10:
            vol_ratio[i] = volume[i] / vol_avg[i]
    
    return vol_ratio

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align HTF RSI for momentum
    rsi_1d_raw = calculate_rsi(df_1d['close'].values, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_raw)
    
    # Calculate primary (6h) indicators
    zscore = calculate_bollinger_zscore(close, period=20)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # Cooldown counter to reduce trade frequency
    cooldown = 0
    COOLDOWN_PERIOD = 10  # Minimum bars between trades
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(zscore[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Cooldown check
        if cooldown > 0:
            cooldown -= 1
        
        # === WEEKLY TREND BIAS (asymmetric logic) ===
        # Only trade WITH the weekly trend
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === DAILY RSI MOMENTUM ===
        daily_rsi = rsi_1d_aligned[i]
        daily_rsi_bull = not np.isnan(daily_rsi) and daily_rsi > 45.0
        daily_rsi_bear = not np.isnan(daily_rsi) and daily_rsi < 55.0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2  # 20% above average
        
        # === Z-SCORE EXTREMES (mean reversion) ===
        zscore_oversold = zscore[i] < -1.8
        zscore_overbought = zscore[i] > 1.8
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        
        # === ENTRY LOGIC (ASYMMETRIC) ===
        desired_signal = 0.0
        
        # LONG: Only when weekly trend is bull
        if weekly_bull:
            # Mean reversion long: Z-score oversold + daily RSI not too weak + volume
            if zscore_oversold and daily_rsi_bull and cooldown == 0:
                if above_sma50:  # Extra filter for strong setups
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: Only when weekly trend is bear
        elif weekly_bear:
            # Mean reversion short: Z-score overbought + daily RSI not too strong + volume
            if zscore_overbought and daily_rsi_bear and cooldown == 0:
                if not above_sma200:  # Extra filter for strong setups
                    desired_signal = -SIZE_STRONG
                else:
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
                # New position or flip - reset cooldown
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                cooldown = COOLDOWN_PERIOD
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