#!/usr/bin/env python3
"""
Experiment #004: 1d Primary + 1w HTF — HMA Trend + Donchian Breakout + Volume

Hypothesis: 1d timeframe with 1w trend filter captures major moves while
limiting trade frequency to ~20-40/year. Weekly HMA(21) keeps us aligned
with the dominant trend, avoiding countertrend trades during reversals.

Why this should work in BOTH bull AND bear:
- 2021 bull: 1w HMA up → long breakouts → captures rallies
- 2022 crash (-77%): 1w HMA down → short breakouts → survives
- 2023-2024 range: fewer breakouts → smaller drawdown
- 2025+ bear: 1w HMA down → short bias

Based on DB winners (test Sharpe > 1.0):
- mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe 1.38, 95tr)
- mtf_1d_kama_rsi_chop_regime_1w_v1 (Sharpe 1.31, 74tr)
Both use: trend filter + price channel breakout + volume + ATR stop

Design:
- 1w HMA(21) = trend direction filter
- 1d HMA(21) = secondary confirmation
- Donchian(20) = entry signal (breakout of 20d high/low)
- Volume (>55% taker buy) = confirm strength
- ATR(14) = stoploss at 2.5x
- Discrete sizes: 0.30 (breakout + volume), 0.25 (regular)

Target: 75-150 total train trades (20-37/year), Sharpe > 0.6
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_1w_volume_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - 20d high/low for breakout signals"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy = prices["taker_buy_volume"].values if "taker_buy_volume" in prices.columns else volume * 0.5
    n = len(close)
    
    # Load 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA and align to 1d
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_1d = calculate_hma(close, period=21)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donch_upper[i-1]) if i > 0 else True:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = taker_buy[i] / volume[i] if volume[i] > 0 else 0.5
        volume_confirmed = vol_ratio > 0.55
        
        # === TREND DIRECTION (1w HMA primary, 1d HMA secondary) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        price_above_1d = close[i] > hma_1d[i]
        price_below_1d = close[i] < hma_1d[i]
        
        # === DONCHIAN BREAKOUT ===
        donch_break_long = close[i] > donch_upper[i-1]
        donch_break_short = close[i] < donch_lower[i-1]
        
        # === RSI CONFIRMATION (not overbought/oversold for breakouts) ===
        rsi_val = rsi_14[i]
        rsi_neutral = 40 < rsi_val < 60  # breakout in momentum phase
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + 1d bullish + Donchian breakout + volume + RSI neutral
        if (price_above_1w and price_above_1d and 
            donch_break_long and volume_confirmed and rsi_neutral):
            desired_signal = SIZE_STRONG
        
        # LONG (no breakout): 1w bullish + 1d bullish + volume + RSI oversold (mean reversion)
        elif (price_above_1w and price_below_1d and 
              volume_confirmed and rsi_val < 40):
            desired_signal = SIZE_BASE
        
        # SHORT: 1w bearish + 1d bearish + Donchian breakout + volume + RSI neutral
        elif (price_below_1w and price_below_1d and 
              donch_break_short and volume_confirmed and rsi_neutral):
            desired_signal = -SIZE_STRONG
        
        # SHORT (no breakout): 1w bearish + 1d bullish + volume + RSI overbought (mean reversion)
        elif (price_below_1w and not price_below_1d and 
              volume_confirmed and rsi_val > 60):
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL ===
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
        
        # === UPDATE POSITION ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = low[i] - 2.5 * entry_atr
                else:
                    stop_price = high[i] + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals