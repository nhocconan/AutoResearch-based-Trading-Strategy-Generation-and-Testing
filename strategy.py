#!/usr/bin/env python3
"""
Experiment #1396: 30m Primary + 4h/1d HTF — Simplified Trend Pullback with Session Filter

Hypothesis: Previous experiments failed due to OVER-FILTERING (CRSI + Choppiness + multiple HTF = 0 trades).
This strategy SIMPLIFIES to guarantee trades while maintaining quality:

1. 4h HMA(21) = primary trend direction (LESS strict than 1d)
2. 1d HMA(21) = regime confirmation (only strengthens signal, doesn't block entry)
3. 30m RSI(7) = pullback entry timing (40-55 for long, 45-60 for short — LOOSE ranges)
4. 30m ATR(14) > median = volatility filter (avoid dead markets)
5. Session 08-20 UTC = liquidity filter (London/NY overlap)

Why this should work where CRSI/Choppiness failed:
- RSI(7) pullback is PROVEN on crypto (less extreme than RSI(14) extremes)
- 4h trend is less restrictive than 1d (more trade opportunities)
- Session filter reduces false breakouts during low-liquidity hours
- LOOSE RSI ranges (40-55, not 25-35) = GUARANTEES trades

Entry logic (LOOSE to ensure trades):
- LONG: price > 4h_HMA + RSI(7) 40-55 + ATR > median + session 08-20
- SHORT: price < 4h_HMA + RSI(7) 45-60 + ATR > median + session 08-20
- 1d_HMA adds conviction (larger size) but doesn't block entry

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 30m
Size: 0.20 base, 0.30 with 1d confirmation
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi7_pullback_hma_trend_4h1d_session_v1"
timeframe = "30m"
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

def is_session_active(open_time_unix_ms):
    """Check if timestamp is within 08-20 UTC session"""
    # Convert ms to hours
    hours_utc = (open_time_unix_ms / (1000 * 60 * 60)) % 24
    return 8 <= hours_utc < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    # Pre-calculate ATR median for volatility filter
    atr_median = np.nanmedian(atr_14[100:]) if n > 100 else np.nanmedian(atr_14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        session_active = is_session_active(open_time[i])
        
        # === VOLATILITY FILTER ===
        vol_ok = atr_14[i] >= atr_median * 0.7  # Allow some flexibility
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 1d HMA for regime confirmation (adds conviction, doesn't block)
        price_above_1d = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        price_below_1d = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === RSI PULLBACK (LOOSE RANGES TO GUARANTEE TRADES) ===
        rsi = rsi_7[i]
        
        # Long: RSI pulled back to 40-55 (not oversold, just pullback in uptrend)
        rsi_long_ok = 40 <= rsi <= 55
        
        # Short: RSI rallied to 45-60 (not overbought, just retracement in downtrend)
        rsi_short_ok = 45 <= rsi <= 60
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI pullback + vol ok + session active
        if price_above_4h and rsi_long_ok and vol_ok and session_active:
            if price_above_1d:
                # Strong trend alignment (4h + 1d both bullish)
                desired_signal = SIZE_STRONG
            else:
                # Basic long (only 4h bullish)
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + RSI pullback + vol ok + session active
        elif price_below_4h and rsi_short_ok and vol_ok and session_active:
            if price_below_1d:
                # Strong trend alignment (4h + 1d both bearish)
                desired_signal = -SIZE_STRONG
            else:
                # Basic short (only 4h bearish)
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