#!/usr/bin/env python3
"""
Experiment #1429: 15m Primary + 1h/1d HTF — Selective Mean Reversion with Volatility Filter

Hypothesis: 15m has failed in 3 prior experiments (#1417, #1421, #1425) due to:
1. Too many trades (>300/year) → fee drag destroys PnL
2. No volatility regime filter → trades during chop = losses
3. Simple HMA+RSI not selective enough

NEW approach for 15m success:
1. 1d HMA(21) for major trend bias — ONLY trade in HTF trend direction
2. 1h ATR ratio (ATR7/ATR30) for volatility regime — only trade when vol elevated (>1.5)
3. 15m RSI(7) extremes for entry timing — oversold bounce in uptrend, overbought fade in downtrend
4. Session filter — prefer UTC 00-12 (London+NY overlap, higher volume)
5. Volume confirmation — current volume > 20-bar average
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)
7. Tight stoploss: 2.0x ATR trailing

Why this should work where prior 15m failed:
- 3+ confluence filters = ~50-80 trades/year (not 300+)
- Volatility filter avoids choppy periods (biggest 15m killer)
- HTF trend filter prevents counter-trend losses
- Session filter captures high-liquidity periods

Target: Sharpe>0.5, trades>=30 train, trades>=3 test, DD>-35%, trades/year<100
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller than 4h due to higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_vol_rsi_session_1h1d_v1"
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

def calculate_sma(series, period):
    """Simple Moving Average"""
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        window = series[i - period + 1:i + 1]
        if not np.any(np.isnan(window)):
            result[i] = np.mean(window)
    
    return result

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    import datetime
    return datetime.datetime.utcfromtimestamp(open_time / 1000).hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1h ATR ratio for volatility regime
    atr_1h_7_raw = calculate_atr(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, period=7)
    atr_1h_30_raw = calculate_atr(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, period=30)
    atr_1h_7_aligned = align_htf_to_ltf(prices, df_1h, atr_1h_7_raw)
    atr_1h_30_aligned = align_htf_to_ltf(prices, df_1h, atr_1h_30_raw)
    
    # Calculate 15m indicators
    hma_15m_16 = calculate_hma(close, period=16)
    hma_15m_48 = calculate_hma(close, period=48)
    atr_15m_14 = calculate_atr(high, low, close, period=14)
    rsi_15m_7 = calculate_rsi(close, period=7)
    rsi_15m_14 = calculate_rsi(close, period=14)
    vol_sma_20 = calculate_sma(volume, period=20)
    
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
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_15m_14[i]) or atr_15m_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_15m_7[i]) or np.isnan(hma_15m_16[i]) or np.isnan(hma_15m_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_1h_7_aligned[i]) or np.isnan(atr_1h_30_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY REGIME FILTER (1h ATR ratio) ===
        atr_ratio = atr_1h_7_aligned[i] / (atr_1h_30_aligned[i] + 1e-10)
        vol_elevated = atr_ratio > 1.3  # Only trade when vol is elevated
        
        # === SESSION FILTER (UTC 00-12 for London/NY overlap) ===
        utc_hour = get_utc_hour(open_time[i])
        session_ok = (utc_hour >= 0 and utc_hour <= 12)
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 1.2 * vol_sma_20[i]
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 15m HMA CROSSOVER (short-term momentum) ===
        hma_bullish = hma_15m_16[i] > hma_15m_48[i]
        hma_bearish = hma_15m_16[i] < hma_15m_48[i]
        
        # === RSI EXTREMES (15m RSI7 for entry timing) ===
        rsi_7 = rsi_15m_7[i]
        rsi_oversold = rsi_7 < 25
        rsi_overbought = rsi_7 > 75
        
        # === ENTRY LOGIC (VERY SELECTIVE - 4+ confluence) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + vol elevated + session OK + volume OK + RSI oversold
        if price_above_1d and vol_elevated and session_ok and volume_ok and rsi_oversold:
            # Extra confirmation: 15m HMA turning bullish or already bullish
            if hma_bullish:
                desired_signal = SIZE_STRONG
            elif hma_15m_16[i] > hma_15m_16[i-1]:  # HMA rising
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + vol elevated + session OK + volume OK + RSI overbought
        elif price_below_1d and vol_elevated and session_ok and volume_ok and rsi_overbought:
            # Extra confirmation: 15m HMA turning bearish or already bearish
            if hma_bearish:
                desired_signal = -SIZE_STRONG
            elif hma_15m_16[i] < hma_15m_16[i-1]:  # HMA falling
                desired_signal = -SIZE_BASE
        
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
                entry_atr = atr_15m_14[i]
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