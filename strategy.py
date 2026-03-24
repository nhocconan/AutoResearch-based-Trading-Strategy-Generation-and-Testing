#!/usr/bin/env python3
"""
Experiment #125: 15m Primary + 4h/1d HTF — RSI Mean Reversion with Trend Bias

Hypothesis: 15m timeframe needs VERY selective entries to avoid fee drag (>100 trades/yr kills Sharpe).
Strategy combines:
- 4h HMA(21) for intermediate trend bias (not too restrictive like 1d)
- 1d HMA(50) for major regime filter (avoid counter-trend in strong regimes)
- 15m RSI(7) for oversold/overbought entries (faster than RSI(14))
- Volume confirmation (>1.5x 20-bar avg) to filter false signals
- Session filter (00-12 UTC) for London/NY overlap liquidity
- Donchian(20) breakout confirmation for entry timing

Key design choices:
- Timeframe: 15m (target 40-100 trades/year with strict filters)
- HTF: 4h HMA for trend, 1d HMA for regime
- Entry: RSI(7)<25 in uptrend OR RSI(7)>75 in downtrend
- Position size: 0.18 (smaller for 15m frequency, reduces fee impact)
- Stoploss: 2.0x ATR trailing (tighter for 15m swings)
- Session: prefer 00-12 UTC (crypto high-liquidity window)

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi7_hma_vol_session_4h1d_v1"
timeframe = "15m"
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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    volume_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return volume_sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    import datetime
    dt = datetime.datetime.utcfromtimestamp(open_time / 1000.0)
    return dt.hour

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
    
    # Calculate and align 4h HMA for intermediate trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major regime filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (15m) indicators
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    volume_sma = calculate_volume_sma(volume, period=20)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # 15m HMA for local trend
    hma_15m = calculate_hma(close, period=13)
    
    signals = np.zeros(n)
    SIZE = 0.18  # 18% position size (smaller for 15m frequency)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(volume_sma[i]) or volume_sma[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
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
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === MAJOR REGIME (1d HMA) ===
        # Avoid counter-trend trades in strong regimes
        major_bull = close[i] > hma_1d_aligned[i]
        major_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 1.5 * volume_sma[i]
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        utc_hour = get_utc_hour(open_time[i])
        session_ok = (utc_hour >= 0 and utc_hour < 12)
        
        # === RSI EXTREMES (15m) ===
        rsi_oversold = rsi[i] < 25.0  # Long entry
        rsi_overbought = rsi[i] > 75.0  # Short entry
        
        # === DONCHIAN CONFIRMATION ===
        # Long: price near Donchian lower (support)
        # Short: price near Donchian upper (resistance)
        donchian_range = donchian_upper[i] - donchian_lower[i]
        if donchian_range > 1e-10:
            price_position = (close[i] - donchian_lower[i]) / donchian_range
            near_lower = price_position < 0.20
            near_upper = price_position > 0.80
        else:
            near_lower = False
            near_upper = False
        
        # === 15m HMA LOCAL TREND ===
        hma_15m_bull = close[i] > hma_15m[i]
        hma_15m_bear = close[i] < hma_15m[i]
        
        # === DESIRED SIGNAL (Confluence Logic) ===
        desired_signal = 0.0
        
        # LONG: RSI oversold + 4h bull + volume + (session OR near_lower)
        if rsi_oversold and htf_bull and volume_ok:
            # Need at least 2 of: session_ok, near_lower, hma_15m_bull, major_bull
            confluence_count = 0
            if session_ok:
                confluence_count += 1
            if near_lower:
                confluence_count += 1
            if hma_15m_bull:
                confluence_count += 1
            if major_bull:
                confluence_count += 1
            
            if confluence_count >= 2:
                desired_signal = SIZE
        
        # SHORT: RSI overbought + 4h bear + volume + (session OR near_upper)
        elif rsi_overbought and htf_bear and volume_ok:
            # Need at least 2 of: session_ok, near_upper, hma_15m_bear, major_bear
            confluence_count = 0
            if session_ok:
                confluence_count += 1
            if near_upper:
                confluence_count += 1
            if hma_15m_bear:
                confluence_count += 1
            if major_bear:
                confluence_count += 1
            
            if confluence_count >= 2:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals