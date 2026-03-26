#!/usr/bin/env python3
"""
Experiment #023: 4h Camarilla S3/R3 Bounce with Choppiness Regime

HYPOTHESIS: Camarilla S3/R3 are mathematically derived support/resistance
levels where price historically reverses. Previous Camarilla attempts failed
due to TOO LOOSE entries (all bounces). The KEY INSIGHT from DB winner:
use CHOPPINESS INDEX as a regime filter. In choppy markets (CHOP > 61.8),
mean reversion works. In trending markets (CHOP < 38.2), require momentum.

TIMEFRAME: 4h primary
HTF: 1d for trend confirmation
TARGET: 75-150 total trades over 4 years (19-38/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(close, high, low):
    """
    Calculate Camarilla pivot levels (S3, S4, R3, R4)
    Classic Camarilla equations:
    - R4 = C + (H - L) * 1.1 / 2
    - R3 = C + (H - L) * 1.1 / 6
    - S3 = C - (H - L) * 1.1 / 6
    - S4 = C - (H - L) * 1.1 / 2
    """
    n = len(close)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        c = close[i]
        h = high[i]
        l = low[i]
        range_hl = h - l
        
        r4[i] = c + range_hl * 0.55  # 1.1/2
        r3[i] = c + range_hl * 0.1833  # 1.1/6
        s3[i] = c - range_hl * 0.1833
        s4[i] = c - range_hl * 0.55
    
    return s3, s4, r3, r4

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    - CHOP > 61.8 = choppy/range market (mean reversion works)
    - CHOP < 38.2 = trending market (momentum works)
    Formula: 100 * LOG10(SUM(ATR(1), period) / (HHV(high, period) - LLV(low, period))) / LOG10(period)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    # ATR(1) = True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        # Sum of ATR(1) over period
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        hl_range = hh - ll
        
        if hl_range > 1e-10:
            log_ratio = np.log10(atr_sum / hl_range)
            log_period = np.log10(period)
            chop[i] = 100 * log_ratio / log_period
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """RSI with min_periods"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return (100 - (100 / (1 + rs))).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d close for trend direction
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d HMA(21) for trend
    def calc_hma(series, period):
        n = len(series)
        half = max(1, period // 2)
        sqrt_n = int(np.sqrt(period))
        weights = np.arange(1, period + 1, dtype=np.float64)
        w_weight = np.sum(weights)
        
        wma_half = pd.Series(series).rolling(window=half, min_periods=half).apply(
            lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
        ).values
        wma_full = pd.Series(series).rolling(window=period, min_periods=period).apply(
            lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
        ).values
        
        diff = 2 * wma_half - wma_full
        hma = pd.Series(diff).rolling(window=sqrt_n, min_periods=sqrt_n).apply(
            lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
        ).values
        return hma
    
    hma_1d = calc_hma(close_1d, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === Calculate local 4h indicators ===
    # Camarilla levels
    s3, s4, r3, r4 = calculate_camarilla(close, high, low)
    
    # Choppiness Index
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, period=14)
    
    # RSI
    rsi = calculate_rsi(close, period=14)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Check data availability
        if np.isnan(atr[i]) or atr[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s3[i]) or np.isnan(r3[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Current values
        c = close[i]
        h = high[i]
        l = low[i]
        chop_val = chop[i]
        rsi_val = rsi[i]
        vol_r = vol_ratio[i]
        
        # Choppiness regime
        is_choppy = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        # 1d trend (aligned to current 4h bar)
        hma_trend = hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else c
        bull_trend = c > hma_trend
        
        # Volume confirmation (1.5x average)
        vol_confirm = vol_r > 1.5
        
        desired_signal = 0.0
        
        if not in_position:
            # === CHOPPY REGIME: Mean Reversion at S3/R3 ===
            if is_choppy:
                # LONG: Price bounces at S3
                if c <= s3[i] * 1.005:  # Within 0.5% of S3
                    if vol_confirm:
                        desired_signal = SIZE
                
                # SHORT: Price bounces at R3
                if c >= r3[i] * 0.995:  # Within 0.5% of R3
                    if vol_confirm:
                        desired_signal = -SIZE
            
            # === TRENDING REGIME: Momentum with Camarilla confirmation ===
            elif is_trended:
                # In uptrend: wait for pullback to S3/S4, then breakout above
                if bull_trend:
                    # Long entry: price at S3/S4 with bullish RSI divergence
                    if c <= s3[i] * 1.01 and rsi_val < 50:
                        desired_signal = SIZE
                
                # In downtrend: short rallies to R3/R4
                else:
                    if c >= r3[i] * 0.99 and rsi_val > 50:
                        desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, h)
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if l < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, l)
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if h > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT: When CHOP shifts ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            # Exit long if chop shifts from choppy to trending AND we're at profit
            profit_pct = (c - entry_price) / entry_price
            if is_trending and profit_pct > 0.01:  # 1% profit in trending market
                tp_triggered = True
            # Or RSI overbought
            if rsi_val > 75:
                tp_triggered = True
        
        if in_position and position_side < 0:
            profit_pct = (entry_price - c) / entry_price
            if is_trending and profit_pct > 0.01:
                tp_triggered = True
            # RSI oversold
            if rsi_val < 25:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = c
                entry_atr = atr[i]
                highest_since_entry = h
                lowest_since_entry = l
                entry_bar = i
                if position_side > 0:
                    stop_price = c - 2.5 * atr[i]
                else:
                    stop_price = c + 2.5 * atr[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_bar = 0
        
        signals[i] = desired_signal
    
    return signals