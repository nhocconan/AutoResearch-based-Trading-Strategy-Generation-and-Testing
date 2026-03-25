#!/usr/bin/env python3
"""
Experiment #1269: 15m Primary + 1h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m strategies have failed due to either (a) too many trades → fee drag,
or (b) over-filtering → 0 trades. This strategy uses HTF for DIRECTION and 15m only
for ENTRY TIMING, achieving HTF trade frequency with lower TF execution precision.

Key innovations vs failed 15m strategies (#1257, #1261, #1265):
1. 1d HMA(21) for MAJOR regime bias — only trade with daily trend (critical filter)
2. 1h HMA(21) for INTERMEDIATE trend confirmation — adds confluence without over-filtering
3. 15m RSI(7) for pullback entries — enters on oversold bounce in uptrend, overbought drop in downtrend
4. Session filter — prefer 00-12 UTC (London+NY overlap), reduces low-volume whipsaws
5. ATR(14) 2.5x trailing stop — protects against reversals
6. Discrete sizing (0.0, ±0.15, ±0.20) — minimizes fee churn on signal changes
7. LOOSE RSI thresholds (25/75 not 20/80) — guarantees trades while remaining selective

Why this should work on 15m:
- HTF direction = ~30-60 trades/year (fee-friendly, not 300+)
- 15m entry timing = better fill prices than 1h/4h entries
- 3-tier confluence (1d + 1h + 15m RSI) = high win rate without over-filtering
- Session filter = avoids Asia session low-volume traps

Entry logic:
- LONG: 1d_HMA bullish + 1h_HMA rising + 15m_RSI(7) < 30 + session 00-12 UTC
- SHORT: 1d_HMA bearish + 1h_HMA falling + 15m_RSI(7) > 70 + session 00-12 UTC

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%, trades/year <100
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_trend_rsi_pullback_session_1h1d_v1"
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
    
    delta = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    # Also calculate 15m HMA for local trend confirmation
    hma_15m = calculate_hma(close, period=21)
    
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
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        utc_hour = get_utc_hour(open_time[i])
        is_preferred_session = (utc_hour >= 0 and utc_hour < 12)
        
        # === TREND DIRECTION (1d HMA bias + 1h HMA slope) ===
        # 1d HMA bias (major regime)
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 1h HMA slope (intermediate trend - compare to 3 bars ago)
        hma_1h_slope = 0.0
        if i >= 3 and not np.isnan(hma_1h_aligned[i-3]):
            hma_1h_slope = hma_1h_aligned[i] - hma_1h_aligned[i-3]
        
        hma_1h_rising = hma_1h_slope > 0
        hma_1h_falling = hma_1h_slope < 0
        
        # 15m price vs 15m HMA for local confirmation
        price_above_15m = close[i] > hma_15m[i]
        price_below_15m = close[i] < hma_15m[i]
        
        # === RSI PULLBACK (entry timing) ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 30  # Loose threshold to guarantee trades
        rsi_overbought = rsi > 70
        
        # === ENTRY LOGIC (3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 1h rising + RSI oversold + (preferred session OR strong setup)
        if price_above_1d and hma_1h_rising and rsi_oversold:
            if is_preferred_session or (price_above_15m and rsi < 25):
                if rsi < 25 and price_above_15m:
                    desired_signal = SIZE_STRONG  # Strong setup
                else:
                    desired_signal = SIZE_BASE  # Basic setup
        
        # SHORT: 1d bearish + 1h falling + RSI overbought + (preferred session OR strong setup)
        elif price_below_1d and hma_1h_falling and rsi_overbought:
            if is_preferred_session or (price_below_15m and rsi > 75):
                if rsi > 75 and price_below_15m:
                    desired_signal = -SIZE_STRONG  # Strong setup
                else:
                    desired_signal = -SIZE_BASE  # Basic setup
        
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