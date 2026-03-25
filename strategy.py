#!/usr/bin/env python3
"""
Experiment #1377: 15m Primary + 4h/12h HTF — Session-Filtered Trend Pullback

Hypothesis: 15m timeframe is underexplored (ZERO successful experiments). This strategy combines:
1. 4h HMA(21) for major trend bias (avoid counter-trend trades)
2. 12h ADX(14) for regime filter (only trade when ADX>20 = trending)
3. 15m RSI(7) for entry timing (oversold bounce in uptrend, overbought fade in downtrend)
4. UTC session filter (00-12 = London/NY overlap, highest volume)
5. Volume confirmation (current volume > 20-bar MA volume)
6. ATR trailing stoploss (2.5x ATR)

Why this should work where 15m failed before:
- 4h/12h HTF filters reduce trade frequency to 40-100/year (fee-friendly)
- Session filter avoids low-volume Asia session whipsaws
- RSI(7) is faster than RSI(14), catches intraday pullbacks
- Volume confirmation ensures breakout has participation
- Discrete sizing (0.15/0.20/0.25) minimizes fee churn on signal changes

Entry logic:
- LONG: 4h_HMA bullish + 12h_ADX>20 + 15m_RSI(7)<30 + volume>vol_ma + session 00-12 UTC
- SHORT: 4h_HMA bearish + 12h_ADX>20 + 15m_RSI(7)>70 + volume>vol_ma + session 00-12 UTC

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete (smaller for 15m frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi7_pullback_hma_adx_session_4h12h_v1"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    adx = np.full(n, np.nan)
    
    plus_sum = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_sum = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_sum = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if tr_sum[i] > 0:
            plus_di[i] = 100 * plus_sum[i] / tr_sum[i]
            minus_di[i] = 100 * minus_sum[i] / tr_sum[i]
    
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period*2-1:] = adx_raw[period*2-1:]
    
    return adx

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

def calculate_volume_ma(volume, period=20):
    """Volume moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_utc_hour(prices, idx):
    """Extract UTC hour from open_time timestamp"""
    open_time = prices['open_time'].iloc[idx]
    # open_time is in milliseconds since epoch
    ts = pd.Timestamp(open_time, unit='ms', tz='UTC')
    return ts.hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing - smaller for 15m frequency
    SIZE_WEAK = 0.15
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (12h ADX - trending only) ===
        adx_12h = adx_12h_aligned[i]
        trending_regime = adx_12h > 20  # ADX > 20 = trending market
        
        # === SESSION FILTER (UTC 00-12 = London/NY overlap) ===
        utc_hour = get_utc_hour(prices, i)
        high_volume_session = (utc_hour >= 0 and utc_hour < 12)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_ma_20[i]
        
        # === RSI EXTREMES (fast RSI for intraday) ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + ADX trending + RSI oversold + volume + session
        long_confluence = sum([price_above_4h, trending_regime, rsi_oversold, volume_confirmed, high_volume_session])
        
        if long_confluence >= 4:  # Need 4 of 5 conditions
            if price_above_4h and trending_regime and rsi_oversold:
                # Base size
                base_size = SIZE_BASE
                
                # Add volume/session conviction
                if volume_confirmed and high_volume_session:
                    base_size = SIZE_STRONG
                elif volume_confirmed or high_volume_session:
                    base_size = SIZE_BASE
                else:
                    base_size = SIZE_WEAK
                
                desired_signal = base_size
        
        # SHORT: 4h bearish + ADX trending + RSI overbought + volume + session
        short_confluence = sum([price_below_4h, trending_regime, rsi_overbought, volume_confirmed, high_volume_session])
        
        if short_confluence >= 4:  # Need 4 of 5 conditions
            if price_below_4h and trending_regime and rsi_overbought:
                # Base size
                base_size = SIZE_BASE
                
                # Add volume/session conviction
                if volume_confirmed and high_volume_session:
                    base_size = SIZE_STRONG
                elif volume_confirmed or high_volume_session:
                    base_size = SIZE_BASE
                else:
                    base_size = SIZE_WEAK
                
                desired_signal = -base_size
        
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
        elif abs(desired_signal) >= SIZE_WEAK * 0.9:
            if desired_signal > 0:
                final_signal = SIZE_WEAK
            else:
                final_signal = -SIZE_WEAK
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