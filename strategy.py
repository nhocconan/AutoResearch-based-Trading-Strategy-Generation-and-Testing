#!/usr/bin/env python3
"""
Experiment #025: 12h Simple Donchian + HMA Trend + Volume Spike

HYPOTHESIS: Winners in DB use Donchian(20) + trend + volume (75-95 trades, Sharpe 1.3-1.5).
Previous 12h TRIX+Camarilla overtraded (172 trades) and got negative Sharpe.
Camarilla was noise. Simpler = fewer trades = less fee drag = better generalization.

WHY 12h: Natural institutional timeframe, captures multi-day moves.
Donchian(20) on 12h = 10-day breakout window.
HMA(16) provides fast trend confirmation without RSI/smoothing lag.
Volume spike confirms smart money.

REGIME: Skip when choppy (CHOP > 61.8).
EXIT: Trailing ATR stop (2.5x) to lock profits.

TARGET: 75-125 total trades over 4 years.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simple_donchian_hma_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=16):
    """Hull Moving Average - faster than EMA, less lag than SMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA of period
    wma_full = pd.Series(close).rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1)), raw=True
    ).values
    
    # WMA of half period
    wma_half = pd.Series(close).rolling(window=half, min_periods=half).apply(
        lambda x: np.sum(x * np.arange(1, half + 1)) / np.sum(np.arange(1, half + 1)), raw=True
    ).values
    
    # HMA = WMA(2*WMA_half - WMA_full) with sqrt period
    raw_hma = 2 * np.where(np.isfinite(wma_half), wma_half, 0) - np.where(np.isfinite(wma_full), wma_full, 0)
    
    hma = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(raw_hma[i]):
            continue
        # Rolling window WMA of the raw HMA
        start = max(0, i - sqrt_n + 1)
        end = i + 1
        window_vals = raw_hma[start:end]
        weights = np.arange(1, len(window_vals) + 1)
        if len(window_vals) == sqrt_n:
            hma[i] = np.sum(window_vals * weights) / np.sum(weights)
    
    return hma

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values > 61.8 = choppy/range, < 38.2 = trending"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(period))
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - use shift(1) to avoid look-ahead"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for HTF trend confirmation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Local 12h indicators ===
    hma_16 = calculate_hma(close, period=16)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 200  # Need 200 for HMA + HTF alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(hma_16[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND: HMA direction + HTF EMA ===
        hma_bullish = close[i] > hma_16[i]
        hma_bearish = close[i] < hma_16[i]
        
        # HTF trend
        above_htf_ema = close[i] > ema_50_aligned[i]
        below_htf_ema = close[i] < ema_50_aligned[i]
        
        # === CHOPPINESS REGIME FILTER ===
        chop = chop_14[i]
        is_choppy = chop > 61.8 if not np.isnan(chop) else False
        
        # === DONCHIAN BREAKOUT (shift(1) to avoid look-ahead) ===
        donchian_broken_up = close[i] > donchian_up[i - 1] if i > 0 else False
        donchian_broken_down = close[i] < donchian_lo[i - 1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and not is_choppy:
            # === LONG ENTRY ===
            # HMA bullish + above HTF EMA + Donchian breakout + volume spike
            if hma_bullish and above_htf_ema and donchian_broken_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # HMA bearish + below HTF EMA + Donchian breakdown + volume spike
            if hma_bearish and below_htf_ema and donchian_broken_down and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (trailing 2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Long stop: price crosses below entry - 2.5 ATR
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                # Exit if HMA flips bearish while in HTF downtrend
                if hma_bearish and below_htf_ema and (i - entry_bar) >= 2:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Short stop: price crosses above entry + 2.5 ATR
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                # Exit if HMA flips bullish while in HTF uptrend
                if hma_bullish and above_htf_ema and (i - entry_bar) >= 2:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals