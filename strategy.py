#!/usr/bin/env python3
"""
Experiment #1208: 4h Primary + 12h/1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After analyzing 960+ failures, the key insight is that BREAKOUT strategies
generate consistent trades (unlike mean-reversion which often gives 0 trades). This
combines:

1. 12h HMA(21) for primary trend direction (price above = bullish bias)
2. 1d HMA(21) for stronger confirmation (optional boost to position size)
3. Donchian(20) breakout for entry trigger (guarantees trades on volatility)
4. RSI(14) filter to avoid exhausted moves (RSI < 70 for longs, > 30 for shorts)
5. ADX(14) > 18 to confirm momentum (loose threshold to ensure trades)
6. ATR(14) 2.5x trailing stop for risk management

Key insight: Donchian breakouts naturally generate 20-50 trades/year on 4h timeframe.
The HMA filter prevents counter-trend breakouts (which fail in 2022 crash).
RSI filter avoids entering at exhaustion (breakout + RSI>80 = trap).

Entry logic (designed to guarantee trades):
- LONG: price > 12h_HMA AND Donchian breakout high AND RSI < 70 AND ADX > 18
- SHORT: price < 12h_HMA AND Donchian breakout low AND RSI > 30 AND ADX > 18

Why this should work:
- 4h timeframe = natural 20-50 trades/year (fee-friendly per Rule 10)
- Donchian breakout = guaranteed entries on volatility (no 0-trade risk)
- HMA trend filter = avoids counter-trend traps (2022 crash protection)
- RSI filter = avoids exhaustion entries (breakout traps)
- ADX loose threshold (18 not 25) = ensures trades trigger
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_adx_12h1d_v1"
timeframe = "4h"
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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.divide(plus_di, atr, out=np.zeros_like(plus_di), where=atr != 0) * 100
    minus_di = np.divide(minus_di, atr, out=np.zeros_like(minus_di), where=atr != 0) * 100
    
    dx = np.divide(np.abs(plus_di - minus_di), plus_di + minus_di, 
                   out=np.zeros_like(plus_di), where=(plus_di + minus_di) != 0) * 100
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    return adx

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
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
    min_bars = 100
    
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
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (12h HMA) ===
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # 1d HMA for additional confirmation (optional boost)
        hma_1d_valid = not np.isnan(hma_1d_aligned[i])
        price_above_1d = hma_1d_valid and close[i] > hma_1d_aligned[i]
        price_below_1d = hma_1d_valid and close[i] < hma_1d_aligned[i]
        
        # === BREAKOUT DETECTION ===
        # Long breakout: price breaks above Donchian upper
        long_breakout = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        
        # Short breakout: price breaks below Donchian lower
        short_breakout = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === FILTERS ===
        rsi = rsi_14[i]
        adx = adx_14[i]
        
        # RSI filter: avoid exhaustion (long: RSI < 70, short: RSI > 30)
        rsi_ok_long = rsi < 70.0
        rsi_ok_short = rsi > 30.0
        
        # ADX filter: loose threshold to ensure trades (ADX > 18)
        adx_ok = adx > 18.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Price above 12h HMA + Donchian breakout + RSI < 70 + ADX > 18
        if price_above_12h and long_breakout and rsi_ok_long and adx_ok:
            if price_above_1d:
                desired_signal = SIZE_STRONG  # Strong trend alignment (12h + 1d)
            else:
                desired_signal = SIZE_BASE  # Basic uptrend breakout
        
        # SHORT: Price below 12h HMA + Donchian breakout + RSI > 30 + ADX > 18
        elif price_below_12h and short_breakout and rsi_ok_short and adx_ok:
            if price_below_1d:
                desired_signal = -SIZE_STRONG  # Strong trend alignment (12h + 1d)
            else:
                desired_signal = -SIZE_BASE  # Basic downtrend breakout
        
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