#!/usr/bin/env python3
"""
Experiment #1319: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: The 6h KAMA+ROC strategy (Sharpe=0.447) works because it captures medium-term
trends with momentum confirmation. For 1h, we need to adapt: use 4h/12h for TREND DIRECTION
(proven edge), 1h only for ENTRY TIMING via RSI pullback. Key innovations:

1. 4h HMA(21) slope for primary trend direction (smoother than 1h, less lag than 12h)
2. 12h HMA(21) for major regime bias (only trade with daily trend)
3. 1h RSI(7) pullback entries (not extreme <20, but <45 for long in uptrend = more trades)
4. Session filter 08-20 UTC (reduces trades to 40-80/year, focuses on liquid hours)
5. ATR(14) 2.5x trailing stop for risk management
6. LOOSE entry conditions to guarantee 30+ trades (RSI<50 not <30)

Why this should work on 1h:
- 4h/12h HTF filters = same directional edge as 6h strategy
- 1h RSI pullback = catches entries on dips within trend (more trades than 6h)
- Session filter = reduces noise trades during low-liquidity hours
- RSI(7) not RSI(14) = more responsive, catches pullbacks earlier
- Discrete sizing (0.0, ±0.20, ±0.30) = minimal fee churn

Entry logic (LOOSE to guarantee 30+ trades):
- LONG: 4h_HMA rising + 12h_HMA bullish + 1h_RSI(7) < 50 (pullback, not extreme)
- SHORT: 4h_HMA falling + 12h_HMA bearish + 1h_RSI(7) > 50 (pullback, not extreme)
- Session: only 08-20 UTC (Asian+European+early US overlap)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_session_4h12h_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=7):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
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

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    # Also calculate 1h HMA for local trend confirmation
    hma_1h = calculate_hma(close, period=21)
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === TREND DIRECTION (4h HMA slope + 12h HMA bias) ===
        # 4h HMA slope (compare to 3 bars ago for stability)
        hma_4h_slope = 0.0
        if i >= 3 and not np.isnan(hma_4h_aligned[i-3]):
            hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-3]
        
        # 12h HMA bias
        price_above_12h = close[i] > hma_12h_aligned[i]
        price_below_12h = close[i] < hma_12h_aligned[i]
        
        # 1h price vs 1h HMA for local confirmation
        price_above_1h = close[i] > hma_1h[i]
        price_below_1h = close[i] < hma_1h[i]
        
        # === PULLBACK ENTRY (RSI) ===
        rsi = rsi_7[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 4h HMA rising + 12h bullish + RSI pullback (<50, not extreme)
        if in_session and hma_4h_slope > 0 and price_above_12h:
            if rsi < 50 and price_above_1h:
                if rsi < 35:
                    desired_signal = SIZE_STRONG  # Deep pullback
                else:
                    desired_signal = SIZE_BASE  # Shallow pullback
        
        # SHORT: 4h HMA falling + 12h bearish + RSI pullback (>50, not extreme)
        elif in_session and hma_4h_slope < 0 and price_below_12h:
            if rsi > 50 and price_below_1h:
                if rsi > 65:
                    desired_signal = -SIZE_STRONG  # Deep pullback
                else:
                    desired_signal = -SIZE_BASE  # Shallow pullback
        
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