#!/usr/bin/env python3
"""
Experiment #1650: 1h Primary + 4h/1d HTF — Volatility Contraction Breakout

Hypothesis: 1h timeframe with 4h trend bias and 1d regime filter captures optimal
entry timing while maintaining trade quality. Volatility contraction (BB squeeze)
followed by expansion signals high-probability breakouts.

Key design based on failure analysis:
1. 4h HMA for trend direction (proven in best strategies)
2. BB squeeze detection for low-volatility entries
3. Volume confirmation to avoid false breakouts
4. Session filter (08-20 UTC) for liquidity
5. Discrete signal sizes: 0.25 base, 0.30 strong
6. 2.5x ATR trailing stoploss

Entry logic (balanced for 40-80 trades/year):
- LONG: 4h HMA bullish + BB squeeze + volume spike + price breaks BB upper
- SHORT: 4h HMA bearish + BB squeeze + volume spike + price breaks BB lower
- 1d HMA as regime filter (only trade with 1d trend)

Why this might beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- 1h TF = more entries than 6h, better capture of intraday moves
- BB squeeze = proven volatility breakout pattern
- Volume filter = avoids false breakouts (failed in #1641, #1645)
- Session filter = avoids low-liquidity whipsaws

Target: Sharpe>0.6, trades≥30 train, trades≥3 test, DD>-35%
Timeframe: 1h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_bb_squeeze_vol_4h1d_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands with bandwidth for squeeze detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # Bandwidth = (upper - lower) / sma
    bandwidth = np.full(n, np.nan, dtype=np.float64)
    mask = sma != 0
    bandwidth[mask] = (upper[mask] - lower[mask]) / sma[mask]
    
    return upper, sma, lower, bandwidth

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    
    mask = atr != 0
    plus_di[mask] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[mask] / atr[mask]
    minus_di[mask] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[mask] / atr[mask]
    
    dx = np.full(n, np.nan, dtype=np.float64)
    di_sum = plus_di + minus_di
    mask2 = di_sum != 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def calculate_bandwidth_percentile(bandwidth, lookback=100):
    """Percentile rank of bandwidth over lookback period"""
    n = len(bandwidth)
    percentile = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        window = bandwidth[i - lookback:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            rank = np.sum(valid <= bandwidth[i]) / len(valid)
            percentile[i] = rank * 100
    
    return percentile

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds
    hour = (open_time // (1000 * 60 * 60)) % 24
    return hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower, bb_bandwidth = calculate_bollinger(close, period=20, std_mult=2.0)
    bb_bw_percentile = calculate_bandwidth_percentile(bb_bandwidth, lookback=100)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma_20[i]) or np.isnan(bb_bw_percentile[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC for liquidity) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === REGIME DETECTION ===
        # 4h trend direction
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 1d trend direction (regime filter)
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === BB SQUEEZE DETECTION ===
        # Bandwidth at low percentile = squeeze
        is_squeeze = bb_bw_percentile[i] < 30  # Bottom 30% of bandwidth
        
        # === VOLUME SPIKE ===
        vol_ratio = volume[i] / vol_sma_20[i] if vol_sma_20[i] > 0 else 0
        vol_spike = vol_ratio > 1.3  # 30% above average
        
        # === ADX TREND STRENGTH ===
        adx_val = adx_14[i]
        is_trending = adx_val > 20  # Minimum trend strength
        
        # === RSI MOMENTUM ===
        rsi_val = rsi_14[i]
        rsi_bullish = rsi_val > 45 and rsi_val < 70
        rsi_bearish = rsi_val < 55 and rsi_val > 30
        
        # === PRICE POSITION ===
        price_near_upper = close[i] >= bb_upper[i] * 0.995
        price_near_lower = close[i] <= bb_lower[i] * 1.005
        price_above_mid = close[i] > bb_mid[i]
        price_below_mid = close[i] < bb_mid[i]
        
        # === ENTRY LOGIC (balanced for 40-80 trades/year) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + 1d bullish + squeeze + volume + breakout
        if price_above_4h and price_above_1d:
            if is_squeeze and vol_spike and price_near_upper and rsi_bullish:
                desired_signal = SIZE_STRONG if adx_val > 30 else SIZE_BASE
            elif price_above_mid and rsi_bullish and vol_spike and is_trending:
                # Secondary entry: momentum continuation
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + 1d bearish + squeeze + volume + breakdown
        elif price_below_4h and price_below_1d:
            if is_squeeze and vol_spike and price_near_lower and rsi_bearish:
                desired_signal = -SIZE_STRONG if adx_val > 30 else -SIZE_BASE
            elif price_below_mid and rsi_bearish and vol_spike and is_trending:
                # Secondary entry: momentum continuation
                desired_signal = -SIZE_BASE
        
        # === SESSION FILTER APPLICATION ===
        # Only enter during session hours, but hold positions through night
        if not in_session and desired_signal != 0 and not in_position:
            desired_signal = 0.0
        
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals