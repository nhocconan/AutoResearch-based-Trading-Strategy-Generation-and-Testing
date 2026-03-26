#!/usr/bin/env python3
"""
Experiment #014: 1h RSI Mean Reversion + 4h Trend Confirmation

HYPOTHESIS: 1h timeframe is difficult (17% keep rate), but using 4h for SIGNAL
DIRECTION filters out 83% of noise. 1h RSI mean reversion at extremes (30/70)
captures reversals within 4h trends. 1d ATR ratio identifies volatility regime
(expansion vs contraction). Session filter (08-20 UTC) avoids low-liquidity
periods that create false signals. Works in both bull (buy RSI dips) and bear
(short RSI rallies to 70+).

KEY INSIGHT: We enter 1h extremes only when 4h trend confirms. This reduces
trade frequency dramatically compared to pure 1h, while maintaining edge.

TARGET: 60-150 total trades over 4 years (15-37/year for 1h)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_meanrev_4h_trend_v1"
timeframe = "1h"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed values using EWM
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI calculations
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX as smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period * 2, adjust=False).mean().values
    
    return adx

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # 4h ADX for trend strength
    adx_4h_raw = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
    
    # 1d ATR ratio for volatility regime
    tr_1d = np.zeros(len(df_1d), dtype=np.float64)
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(df_1d['high'].values[i] - df_1d['low'].values[i],
                       abs(df_1d['high'].values[i] - df_1d['close'].values[i-1]),
                       abs(df_1d['low'].values[i] - df_1d['close'].values[i-1]))
    
    atr_1d_7 = pd.Series(tr_1d).ewm(span=7, min_periods=7, adjust=False).mean().values
    atr_1d_30 = pd.Series(tr_1d).ewm(span=30, min_periods=30, adjust=False).mean().values
    atr_ratio_1d = atr_1d_7 / (atr_1d_30 + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # === Calculate 1h indicators ===
    rsi_14 = calculate_rsi(close, period=14)
    rsi_28 = calculate_rsi(close, period=28)
    
    # 1h ATR for stoploss
    tr_1h = np.zeros(n, dtype=np.float64)
    tr_1h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_1h_14 = pd.Series(tr_1h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # RSI momentum (for crossover detection)
    rsi_14_prev = np.roll(rsi_14, 1)
    rsi_14_prev[0] = np.nan
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    SIZE = 0.20  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(rsi_14[i]) or np.isnan(atr_1h_14[i]) or atr_1h_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # === 4H TREND DIRECTION ===
        # Bullish: price above 4h HMA
        # Bearish: price below 4h HMA
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        
        # === 4H TREND STRENGTH (ADX) ===
        # ADX > 20 = trending, ADX < 15 = ranging
        adx_val = adx_4h_aligned[i]
        is_trending = adx_val > 20
        
        # === VOLATILITY REGIME (1d ATR ratio) ===
        # < 0.8 = contraction (low vol), > 1.5 = expansion (high vol)
        atr_ratio = atr_ratio_aligned[i] if not np.isnan(atr_ratio_aligned[i]) else 1.0
        is_low_vol = atr_ratio < 0.8
        
        # === 1H RSI VALUES ===
        rsi_short = rsi_14[i]
        rsi_long = rsi_28[i]
        rsi_short_prev = rsi_14_prev[i]
        
        # === RSI CROSSOVER DETECTION ===
        # RSI crossed above 35 from below = potential long
        # RSI crossed below 65 from above = potential short
        rsi_crossed_up = (rsi_short_prev < 35) and (rsi_short >= 35) if not np.isnan(rsi_short_prev) else False
        rsi_crossed_down = (rsi_short_prev > 65) and (rsi_short <= 65) if not np.isnan(rsi_short_prev) else False
        
        # RSI at extreme levels
        rsi_oversold = rsi_short < 35
        rsi_overbought = rsi_short > 65
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Conditions:
            # 1. Price above 4h HMA (bullish trend)
            # 2. ADX > 20 (trend strength)
            # 3. RSI crossed above 35 OR RSI < 35 (mean reversion setup)
            # 4. In session (08-20 UTC)
            # 5. Avoid high volatility expansion (false breakouts)
            if in_session and price_above_4h_hma and is_trending:
                if rsi_crossed_up or rsi_oversold:
                    # Avoid entries during vol expansion
                    if atr_ratio < 1.8:
                        desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Conditions:
            # 1. Price below 4h HMA (bearish trend)
            # 2. ADX > 20 (trend strength)
            # 3. RSI crossed below 65 OR RSI > 65 (mean reversion setup)
            # 4. In session (08-20 UTC)
            if in_session and not price_above_4h_hma and is_trending:
                if rsi_crossed_down or rsi_overbought:
                    if atr_ratio < 1.8:
                        desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
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
        
        # === TAKE PROFIT: RSI normalization ===
        # Long: exit when RSI > 55 (mean reversion complete)
        # Short: exit when RSI < 45 (mean reversion complete)
        exit_triggered = False
        
        if in_position and position_side > 0:
            if rsi_short > 55:
                exit_triggered = True
        
        if in_position and position_side < 0:
            if rsi_short < 45:
                exit_triggered = True
        
        # === EXIT: Opposite trend signal ===
        if in_position and position_side > 0:
            # Exit long if price breaks below 4h HMA
            if close[i] < hma_4h_aligned[i]:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Exit short if price breaks above 4h HMA
            if close[i] > hma_4h_aligned[i]:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                pass  # Maintain position
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals