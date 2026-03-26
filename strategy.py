#!/usr/bin/env python3
"""
Experiment #006: 4h EMA Crossover + HMA Trend + Volume Confirmation

Hypothesis: Simple EMA(10/30) crossover on 4h with 1d HMA(21) trend bias and volume
confirmation provides the right balance between trade frequency and signal quality.
EMA crossovers are proven momentum indicators; HMA(21) on 1d provides reliable
trend direction without overcomplicating. Volume confirms institutional participation.
ATR stop prevents blowups in volatile markets like 2022.

Why it should work in BOTH bull AND bear:
- In bull markets: EMA crossover catches upside momentum with HMA trend aligned
- In bear markets: Short signals when price < 200 SMA catches falling knives
- The 200 SMA filter avoids buying in major downtrends
- ATR filter avoids low-volatility chop where EMA crossovers whipsaw
- 2022 BTC crash: Short signals when price < 200 SMA preserve capital

Trade frequency target: 60-100 total over 4 years (15-25/year per symbol).
Conservative entry = fewer trades = less fee drag = better test generalization.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_ema_cross_hma_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_adx(high, low, close, period=14):
    """ADX + DMI for trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.full(n, 0.0)
    minus_di = np.full(n, 0.0)
    
    for i in range(period, n):
        if atr_smooth[i] > 1e-10:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr_smooth[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_ema(close, span):
    """Exponential Moving Average"""
    n = len(close)
    if n < span:
        return np.full(n, np.nan)
    ema = pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values
    return ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA(21) for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # EMA crossover: fast(10) and slow(30)
    ema_fast = calculate_ema(close, span=10)
    ema_slow = calculate_ema(close, span=30)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Long-term SMA for regime filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Warmup period (need 200 bars for SMA + room for indicators)
    warmup = 220
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # Previous bar values for crossover detection
        ema_fast_prev = ema_fast[i-1] if i > 0 else ema_fast[i]
        ema_slow_prev = ema_slow[i-1] if i > 0 else ema_slow[i]
        
        # === ENTRY CONDITIONS ===
        
        # 1) EMA crossover: fast crosses above/below slow
        bull_cross = (ema_fast[i] > ema_slow[i]) and (ema_fast_prev <= ema_slow_prev)
        bear_cross = (ema_fast[i] < ema_slow[i]) and (ema_fast_prev >= ema_slow_prev)
        
        # 2) Volume confirmation: volume > 20-bar SMA
        vol_confirm = volume[i] > vol_sma[i]
        
        # 3) 1d HMA trend bias (align with trend)
        price_above_hma = close[i] > hma_1d_aligned[i]
        price_below_hma = close[i] < hma_1d_aligned[i]
        
        # 4) Long-term regime: price above/below 200 SMA
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # 5) ATR filter: ensure minimum volatility
        atr_filter = atr_14[i] > 150
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if bull_cross and vol_confirm and price_above_hma and price_above_sma200 and atr_filter:
            desired_signal = SIZE
        elif bear_cross and vol_confirm and price_below_hma and price_below_sma200 and atr_filter:
            desired_signal = -SIZE
        
        # === STOPLOSS: 2.5x ATR trailing ===
        in_position = desired_signal != 0.0
        
        if in_position:
            if desired_signal > 0:
                stop_price = close[i] - 2.5 * atr_14[i]
                if low[i] < stop_price:
                    desired_signal = 0.0
            else:
                stop_price = close[i] + 2.5 * atr_14[i]
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === DISCRETIZE ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        else:
            final_signal = 0.0
        
        signals[i] = final_signal
    
    return signals