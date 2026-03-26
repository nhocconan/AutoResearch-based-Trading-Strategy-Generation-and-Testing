#!/usr/bin/env python3
"""
Experiment #023: 1d KAMA Trend + RSI Momentum + Choppiness Regime

HYPOTHESIS: KAMA (Kaufman Adaptive Moving Average) adapts to volatility, making it 
superior to fixed-period MAs for trend detection. Combined with RSI momentum 
confirmation and Choppiness Index regime filter, this captures trending moves 
while avoiding range-bound whipsaws. The 1d timeframe ensures ~7-25 trades/year, 
minimizing fee drag. Works in bull (long KAMA uptrends) and bear (short KAMA 
downtrends with rallies to KAMA as short opportunities).

TIMEFRAME: 1d primary
HTF: 1w for KAMA trend (confirmed SOLUSDT test Sharpe 1.31 in DB)
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast_ema=2, slow_ema=30):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period + slow_ema:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    direction = np.abs(close[period:] - close[:-period])
    volatility = np.zeros(n - period)
    for i in range(n - period):
        for j in range(period):
            volatility[i] += abs(close[i + j + 1] - close[i + j])
        volatility[i] = max(volatility[i], 1e-10)
    
    er = np.zeros(n)
    er[period:] = direction / volatility
    
    # Smooth constant
    fast_const = 2 / (fast_ema + 1)
    slow_const = 2 / (slow_ema + 1)
    ssc = (er * (fast_const - slow_const) + slow_const) ** 2
    
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + ssc[i - period] * (close[i] - kama[i - 1])
    
    return kama

def calculate_choppiness(close, high, low, period=14):
    """Choppiness Index - measures trendiness vs choppiness"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(period):
            tr = max(high[i - j] - low[i - j], 
                    abs(high[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j],
                    abs(low[i - j] - close[i - j - 1]) if i - j - 1 >= 0 else high[i - j] - low[i - j])
            atr_sum += tr
        
        # Highest - Lowest over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = max(hh - ll, 1e-10)
        
        # Choppiness formula
        chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """RSI with min_periods"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly KAMA for macro trend
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate local 1d indicators
    kama_1d = calculate_kama(close, period=10)
    rsi_1d = calculate_rsi(close, period=14)
    chop_1d = calculate_choppiness(close, high, low, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kama_1d[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_1d[i]) or np.isnan(chop_1d[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (Choppiness) ===
        # CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        is_trending = chop_1d[i] < 50.0  # Use 50 as neutral threshold
        
        # === TREND (1w KAMA alignment) ===
        price_above_1w_kama = close[i] > kama_1w_aligned[i]
        
        # === TREND (1d KAMA direction) ===
        kama_1d_up = kama_1d[i] > kama_1d[i - 1] if i > warmup and not np.isnan(kama_1d[i - 1]) else True
        kama_1d_down = kama_1d[i] < kama_1d[i - 1] if i > warmup and not np.isnan(kama_1d[i - 1]) else False
        
        # === MOMENTUM (RSI) ===
        rsi_val = rsi_1d[i]
        rsi_not_extreme = 30 < rsi_val < 70  # Avoid extremes on entry
        
        # === VOLUME (confirmation) ===
        vol_ok = vol_ratio[i] > 0.8  # At least average volume
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG ENTRY ===
            # Bull trend: price above 1w KAMA + 1d KAMA rising + RSI confirming
            if price_above_1w_kama and kama_1d_up and rsi_val > 45 and rsi_val < 70:
                if is_trending and vol_ok:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Bear trend: price below 1w KAMA + 1d KAMA falling + RSI confirming
            if not price_above_1w_kama and kama_1d_down and rsi_val < 55 and rsi_val > 30:
                if is_trending and vol_ok:
                    desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: KAMA turns down OR RSI drops below 40 OR price breaks below 1w KAMA
            if kama_1d_down:
                exit_triggered = True
            if rsi_val < 40:
                exit_triggered = True
            if close[i] < kama_1w_aligned[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: KAMA turns up OR RSI rises above 60 OR price breaks above 1w KAMA
            if kama_1d_up:
                exit_triggered = True
            if rsi_val > 60:
                exit_triggered = True
            if close[i] > kama_1w_aligned[i]:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_bar = i
            else:
                pass  # Maintain position
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals