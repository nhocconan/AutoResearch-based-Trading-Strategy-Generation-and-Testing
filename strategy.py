#!/usr/bin/env python3
"""
Experiment #1637: 15m Primary + 4h/12h HTF — Simple RSI Pullback with Volume Filter

Hypothesis: 15m timeframe has been untested (0 experiments). Key insight from failures:
15m strategies fail due to TOO STRICT entries (Sharpe=0.000 = 0 trades in #1625, #1629, #1633).

This strategy uses LOOSE entry conditions to guarantee trades:
1. 4h HMA(21) for trend bias (simple, proven in #1618)
2. 15m RSI(7) with LOOSE thresholds: <40/>60 (not <30/>70)
3. 15m EMA(21) as pullback confirmation (price retests EMA in trend direction)
4. Volume spike filter: volume > 1.5x 20-bar average (confirms momentum)
5. Session filter: 00-12 UTC only (London+NY overlap, reduces noise)

Key design choices based on 15m failure analysis:
- LOOSE RSI thresholds to guarantee ≥40 trades/year
- Volume filter NOT too strict (1.5x not 2.0x)
- Session filter reduces whipsaws but doesn't block all entries
- Small position size: 0.15-0.20 (15m has higher frequency)
- 2.0x ATR stoploss (tighter for faster TF)

Why this might beat 6h baseline (Sharpe=0.575):
- 15m catches intraday moves that 6h misses
- RSI(7) more responsive than RSI(14)
- Volume confirmation filters false breakouts
- Session filter avoids Asian session chop

Target: Sharpe>0.6, trades≥40 train, trades≥5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_ema_volume_4h_session_v1"
timeframe = "15m"
leverage = 1.0

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time_array // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 15m indicators
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Fast RSI for 15m
    rsi_14 = calculate_rsi(close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Extract UTC hour for session filter
    utc_hour = get_hour_from_open_time(open_time)
    
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
    min_bars = 60
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC = London+NY overlap) ===
        # Only take new entries during active sessions
        in_session = (utc_hour[i] >= 0) and (utc_hour[i] <= 12)
        
        # === TREND DIRECTION (4h + 12h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # Strong trend: both 4h and 12h agree
        strong_bullish = price_above_4h and price_above_12h
        strong_bearish = price_below_4h and price_below_12h
        
        # === RSI SIGNALS (LOOSE thresholds for 15m) ===
        rsi_7_val = rsi_7[i]
        rsi_14_val = rsi_14[i]
        
        # LOOSE thresholds to guarantee trades
        rsi_oversold = rsi_7_val < 40  # Not <30 (too strict)
        rsi_overbought = rsi_7_val > 60  # Not >70 (too strict)
        rsi_neutral_low = rsi_7_val < 50
        rsi_neutral_high = rsi_7_val > 50
        
        # === EMA PULLBACK CONFIRMATION ===
        # Price pulling back to EMA21 in uptrend
        pullback_to_ema_long = (low[i] <= ema_21[i] * 1.002) and (close[i] > ema_21[i])
        # Price pulling back to EMA21 in downtrend
        pullback_to_ema_short = (high[i] >= ema_21[i] * 0.998) and (close[i] < ema_21[i])
        
        # === VOLUME CONFIRMATION ===
        volume_spike = vol_ratio[i] > 1.5  # 1.5x average (not 2.0x, too strict)
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: Strong bullish trend + RSI oversold + pullback OR volume spike
        if strong_bullish:
            # Entry condition 1: RSI oversold + pullback to EMA
            if rsi_oversold and pullback_to_ema_long:
                desired_signal = SIZE_STRONG if volume_spike else SIZE_BASE
            # Entry condition 2: RSI recovering from oversold (cross above 40)
            elif rsi_7_val > 40 and rsi_7_val < 55 and price_above_4h:
                if i > 0 and not np.isnan(rsi_7[i-1]) and rsi_7[i-1] <= 40:
                    desired_signal = SIZE_BASE
        
        # SHORT: Strong bearish trend + RSI overbought + pullback OR volume spike
        elif strong_bearish:
            # Entry condition 1: RSI overbought + pullback to EMA
            if rsi_overbought and pullback_to_ema_short:
                desired_signal = -SIZE_STRONG if volume_spike else -SIZE_BASE
            # Entry condition 2: RSI rolling over from overbought (cross below 60)
            elif rsi_7_val < 60 and rsi_7_val > 45 and price_below_4h:
                if i > 0 and not np.isnan(rsi_7[i-1]) and rsi_7[i-1] >= 60:
                    desired_signal = -SIZE_BASE
        
        # NEUTRAL/WEAK TREND: Use 4h only + RSI extremes (more trades)
        else:
            # LONG: 4h bullish + RSI very oversold
            if price_above_4h and rsi_7_val < 35:
                desired_signal = SIZE_BASE
            # SHORT: 4h bearish + RSI very overbought
            elif price_below_4h and rsi_7_val > 65:
                desired_signal = -SIZE_BASE
        
        # Only take NEW entries during session hours
        # But allow existing positions to continue
        if desired_signal != 0.0 and not in_position:
            if not in_session:
                desired_signal = 0.0
        
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