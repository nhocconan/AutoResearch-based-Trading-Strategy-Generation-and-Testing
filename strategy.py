#!/usr/bin/env python3
"""
Experiment #1021: 15m Primary + 1h/4h/1d HTF — Session-Filtered Multi-Confluence

Hypothesis: 15m strategies fail due to too many trades (fee drag) and noise.
Solution: Use 4h HMA for trend DIRECTION, 1d Choppiness for REGIME,
15m RSI(7)+Stoch for ENTRY TIMING, session filter (00-12 UTC), volume confirmation.

Key innovations:
1. 4h HMA(21) trend filter: Only long when price>4h_HMA, only short when price<4h_HMA
2. 1d Choppiness(14): >61.8 = range (mean revert entries), <38.2 = trend (breakout entries)
3. 15m RSI(7) + Stoch(14,3,3): Dual momentum confirmation for entry timing
4. Session filter: Only trade 00-12 UTC (London+NY overlap = highest volume)
5. Volume confirmation: Current volume > 1.5x 20-bar average
6. ATR(14) 2.0x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)

Why this should work:
- HTF filters reduce noise and whipsaws (4h trend + 1d regime)
- Session filter avoids low-volume Asian session fakeouts
- Volume confirmation ensures real moves, not noise
- Dual momentum (RSI+Stoch) reduces false entries
- Target: 40-100 trades/year (strict confluence = 3+ conditions)

Entry conditions (LOOSE enough for trades, strict enough for quality):
- LONG: 4h_HMA_bull + (RSI7<35 OR Stoch<20) + volume>1.5x_avg + session_00_12
- SHORT: 4h_HMA_bear + (RSI7>65 OR Stoch>80) + volume>1.5x_avg + session_00_12
- Range regime: Add CRSI extremes for mean reversion
- Trend regime: Add RSI(14)>50/<50 for momentum

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_hma_rsi_stoch_vol_4h1d_v1"
timeframe = "15m"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator %K and %D"""
    n = len(close)
    if n < k_period + d_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    k = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        price_range = highest_high - lowest_low
        if price_range > 1e-10:
            k[i] = 100.0 * (close[i] - lowest_low) / price_range
    
    d = pd.Series(k).ewm(span=d_period, min_periods=d_period, adjust=False).mean().values
    return k, d

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_volume_ma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_1d_raw = calculate_choppiness(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values, 
        period=14
    )
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    stoch_k, stoch_d = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(stoch_k[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = (utc_hour >= 0 and utc_hour < 12)
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0.0
        high_volume = vol_ratio > 1.5
        
        # === HTF TREND FILTER (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (1d Choppiness) ===
        is_choppy = chop_1d_aligned[i] > 55.0  # Range market
        is_trending = chop_1d_aligned[i] < 45.0  # Trend market
        
        # === ENTRY LOGIC (MULTI-CONFLUENCE) ===
        desired_signal = 0.0
        
        # LONG conditions (need 4h bull trend + momentum extreme + volume + session)
        if hma_4h_bull and in_session:
            # Mean reversion in choppy market
            if is_choppy and (rsi_7[i] < 30 or stoch_k[i] < 20):
                if high_volume:
                    desired_signal = SIZE_BASE
                elif rsi_7[i] < 20 or stoch_k[i] < 10:
                    desired_signal = SIZE_STRONG
            
            # Trend continuation in trending market
            elif is_trending and rsi_14[i] > 45 and rsi_14[i] < 70:
                if high_volume:
                    desired_signal = SIZE_BASE
                elif rsi_7[i] < 40 and stoch_k[i] < 30:
                    desired_signal = SIZE_STRONG
        
        # SHORT conditions (need 4h bear trend + momentum extreme + volume + session)
        elif hma_4h_bear and in_session:
            # Mean reversion in choppy market
            if is_choppy and (rsi_7[i] > 70 or stoch_k[i] > 80):
                if high_volume:
                    desired_signal = -SIZE_BASE
                elif rsi_7[i] > 80 or stoch_k[i] > 90:
                    desired_signal = -SIZE_STRONG
            
            # Trend continuation in trending market
            elif is_trending and rsi_14[i] < 55 and rsi_14[i] > 30:
                if high_volume:
                    desired_signal = -SIZE_BASE
                elif rsi_7[i] > 60 and stoch_k[i] > 70:
                    desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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