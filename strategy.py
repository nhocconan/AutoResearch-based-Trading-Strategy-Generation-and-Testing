#!/usr/bin/env python3
"""
Experiment #1645: 15m Primary + 4h/1d HTF — Camarilla Pivot Mean Reversion

Hypothesis: 15m timeframe with Camarilla pivot levels from 1d HTF + 4h trend bias
captures intraday mean-reversion opportunities while respecting higher-timeframe direction.
Camarilla levels (R3/S3) act as natural support/resistance for crypto intraday trading.

Key design choices based on 15m failure analysis (#1637, #1641 = 0 trades):
1. LOOSE RSI thresholds: 35/65 (not 30/70) to guarantee entries
2. Multiple pivot levels: S2/R2 + S3/R3 (not just extremes)
3. Session filter: 00-12 UTC (London/NY overlap) for liquidity
4. 4h HMA trend bias (not 1d only) - more responsive for 15m entries
5. Discrete signal sizes: 0.15 base, 0.20 strong (smaller for 15m frequency)
6. 2.0x ATR trailing stoploss via signal→0

Entry logic (LOOSE to guarantee ≥40 trades/train):
- LONG: 4h HMA bullish + price near S2/S3 + RSI(7)<45 + session 00-12 UTC
- SHORT: 4h HMA bearish + price near R2/R3 + RSI(7)>55 + session 00-12 UTC
- BREAKOUT: price crosses R4/S4 with 4h momentum confirmation

Why 15m can work (if trades generated):
- More entry opportunities than 4h/6h
- Tighter stops = better risk/reward
- Intraday mean-reversion works well in crypto range markets (2022-2024)

Target: Sharpe>0.6, trades≥40 train, trades≥5 test, DD>-35%, trades/year<100
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_rsi_4h1d_session_v1"
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

def calculate_camarilla_pivots(high, low, close, prev_close):
    """
    Camarilla Pivot Points - intraday support/resistance levels
    R3/S3 = mean reversion zones, R4/S4 = breakout levels
    
    H = (High - Low) * 1.1 / 12
    R3 = Close + H * 1.1
    R4 = Close + H * 1.1 / 2
    S3 = Close - H * 1.1
    S4 = Close - H * 1.1 / 2
    """
    n = len(close)
    
    r3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    pivot = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        if not np.isnan(prev_close[i-1]) and not np.isnan(high[i-1]) and not np.isnan(low[i-1]):
            h = high[i-1]
            l = low[i-1]
            c = prev_close[i-1]
            
            range_val = h - l
            if range_val > 1e-10:
                h_val = range_val * 1.1 / 12.0
                
                pivot[i] = (h + l + c) / 3.0
                r3[i] = c + h_val * 1.1
                r4[i] = c + h_val * 1.1 * 2.0
                s3[i] = c - h_val * 1.1
                s4[i] = c - h_val * 1.1 * 2.0
    
    return r3, r4, s3, s4, pivot

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000.0
    hour = (ts_seconds % 86400) / 3600.0
    return int(hour)

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
    
    # Get 1d OHLC for Camarilla pivots (use previous day's data)
    df_1d_shifted = df_1d.shift(1)  # Use completed day's data
    prev_day_high = df_1d_shifted['high'].values
    prev_day_low = df_1d_shifted['low'].values
    prev_day_close = df_1d_shifted['close'].values
    
    # Align 1d pivot data to 15m
    pivots_1d_aligned = align_htf_to_ltf(prices, df_1d, np.arange(len(df_1d)))
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate Camarilla pivots from 1d data
    r3_1d, r4_1d, s3_1d, s4_1d, pivot_1d = calculate_camarilla_pivots(
        prev_day_high, prev_day_low, prev_day_close, prev_day_close
    )
    
    # Align pivot levels to 15m
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
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
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
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
        
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London/NY overlap) ===
        hour = get_session_hour(open_time[i])
        is_prime_session = 0 <= hour <= 12
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === CAMARILLA PIVOT LEVELS ===
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        pivot = pivot_aligned[i]
        
        # Price proximity to pivot levels (within 0.5%)
        near_s3 = abs(close[i] - s3) / s3 < 0.005 if not np.isnan(s3) else False
        near_s4 = abs(close[i] - s4) / s4 < 0.005 if not np.isnan(s4) else False
        near_r3 = abs(close[i] - r3) / r3 < 0.005 if not np.isnan(r3) else False
        near_r4 = abs(close[i] - r4) / r4 < 0.005 if not np.isnan(r4) else False
        
        # Price below S3 (oversold) or above R3 (overbought)
        below_s3 = close[i] < s3 if not np.isnan(s3) else False
        above_r3 = close[i] > r3 if not np.isnan(r3) else False
        below_s2 = close[i] < (pivot - (pivot - s3) * 0.5) if not np.isnan(pivot) and not np.isnan(s3) else False
        above_r2 = close[i] > (pivot + (r3 - pivot) * 0.5) if not np.isnan(pivot) and not np.isnan(r3) else False
        
        # === RSI SIGNALS (LOOSE thresholds for trades) ===
        rsi_val = rsi_7[i]
        rsi_bullish = rsi_val < 45  # LOOSE: not just <30
        rsi_bearish = rsi_val > 55  # LOOSE: not just >70
        rsi_oversold = rsi_val < 35
        rsi_overbought = rsi_val > 65
        
        # === ENTRY LOGIC (LOOSE - must generate ≥40 trades/train) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + near support + RSI bullish + prime session
        if price_above_4h:
            # Mean reversion at S2/S3
            if (near_s3 or near_s4 or below_s3) and rsi_bullish:
                desired_signal = SIZE_STRONG if is_prime_session else SIZE_BASE
            # Breakout above R4 with momentum
            elif close[i] > r4 and rsi_val > 50 and rsi_val < 70:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + near resistance + RSI bearish + prime session
        elif price_below_4h:
            # Mean reversion at R2/R3
            if (near_r3 or near_r4 or above_r3) and rsi_bearish:
                desired_signal = -SIZE_STRONG if is_prime_session else -SIZE_BASE
            # Breakout below S4 with momentum
            elif close[i] < s4 and rsi_val < 50 and rsi_val > 30:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL: RSI extremes only (catches trades when trend unclear)
        else:
            if rsi_oversold and (below_s3 or near_s3):
                desired_signal = SIZE_BASE
            elif rsi_overbought and (above_r3 or near_r3):
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