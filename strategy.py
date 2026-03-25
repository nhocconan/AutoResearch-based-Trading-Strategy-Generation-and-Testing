#!/usr/bin/env python3
"""
Experiment #1149: 15m Primary + 1h/4h HTF — Simple Mean Reversion + Trend Bias

Hypothesis: 15m strategies fail because of OVER-FILTERING (too many regime conditions).
This strategy uses SIMPLE confluence that actually triggers:
1. 1h HMA(21) for trend BIAS (soft filter, not hard requirement)
2. 15m RSI(7) for entry timing (oversold <25 / overbought >75)
3. 15m Bollinger Bands(20,2.0) for mean reversion levels
4. Session filter: 08:00-20:00 UTC (high volume hours for crypto)
5. ATR(14) 2.0x trailing stop for risk management

Why this should work on 15m:
- RSI(7) is faster than RSI(14) — triggers more often on intraday swings
- BB(20,2.0) gives clear mean reversion levels that trigger frequently
- 1h HMA bias is SOFT — we can enter counter-trend at extremes (RSI<15 or >85)
- Session filter reduces low-volume whipsaws (00:00-08:00 UTC)
- Size=0.15-0.20 (smaller for 15m frequency) to control drawdown

Entry conditions (LOOSE to guarantee trades):
- LONG: RSI(7)<25 + price<BB_lower + (1h_HMA_bull OR RSI<15)
- SHORT: RSI(7)>75 + price>BB_upper + (1h_HMA_bear OR RSI>85)

Target: Sharpe>0.4, trades>=50 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete
Trade frequency: 50-100/year (critical for 15m fee management)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_bb_meanrev_1h_bias_v1"
timeframe = "15m"
leverage = 1.0

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with middle, upper, lower"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    middle = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    
    return middle, upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08:00-20:00 UTC high volume hours) ===
        # Convert open_time (ms) to hour
        hour_utc = (open_time[i] // 3600000) % 24
        is_active_session = 8 <= hour_utc <= 20
        
        # === HTF BIAS (1h HMA) ===
        hma_1h_bull = close[i] > hma_1h_aligned[i]
        hma_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === ENTRY LOGIC (MEAN REVERSION WITH TREND BIAS) ===
        desired_signal = 0.0
        
        # Price position relative to BB
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        # RSI extremes
        rsi_oversold = rsi_7[i] < 25.0
        rsi_extreme_oversold = rsi_7[i] < 15.0
        rsi_overbought = rsi_7[i] > 75.0
        rsi_extreme_overbought = rsi_7[i] > 85.0
        
        # LONG entries (mean reversion in oversold territory)
        if is_active_session and price_below_bb and rsi_oversold:
            # Strong long: extreme RSI (can ignore trend bias)
            if rsi_extreme_oversold:
                desired_signal = SIZE_STRONG
            # Normal long: needs trend bias confirmation
            elif hma_1h_bull:
                desired_signal = SIZE_BASE
            # Counter-trend long: very oversold + RSI(14) confirms
            elif rsi_14[i] < 30.0:
                desired_signal = SIZE_BASE
        
        # SHORT entries (mean reversion in overbought territory)
        elif is_active_session and price_above_bb and rsi_overbought:
            # Strong short: extreme RSI (can ignore trend bias)
            if rsi_extreme_overbought:
                desired_signal = -SIZE_STRONG
            # Normal short: needs trend bias confirmation
            elif hma_1h_bear:
                desired_signal = -SIZE_BASE
            # Counter-trend short: very overbought + RSI(14) confirms
            elif rsi_14[i] > 70.0:
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