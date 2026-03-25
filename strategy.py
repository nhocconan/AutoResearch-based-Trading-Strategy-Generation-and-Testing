#!/usr/bin/env python3
"""
Experiment #1101: 15m Primary + 4h/1d HTF — RSI Mean Reversion with Trend Bias

Hypothesis: 15m timeframe needs faster entry signals (RSI(7) not RSI(14)) with 
4h trend filter (not 1d/1w which are too slow). Session filter (00-12 UTC) 
captures London/NY overlap liquidity. Bollinger Band position adds mean-reversion 
context. LOOSE entry conditions (RSI<30/>70 not <20/>80) to guarantee trades.

Key innovations:
1. 4h HMA(21) for trend direction — faster than 1d for 15m entries
2. RSI(7) for entry timing — more responsive than RSI(14) on 15m
3. Bollinger Band %B — enter when price at band extremes (mean reversion)
4. Session filter: 00-12 UTC (London+NY overlap = 60% of crypto volume)
5. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)
6. ATR(14) 2.0x trailing stop — tighter for lower TF

Why this should work on 15m:
- 4h trend filter prevents counter-trend trades (major failure mode)
- RSI(7) < 30 triggers ~2-3x/month per symbol = 30-100 trades/year target
- BB %B < 0.1 or > 0.9 confirms price at extremes (not mid-range noise)
- Session filter avoids low-liquidity Asia-only hours (whipsaw prone)
- Smaller size (0.20) accounts for higher trade frequency fee drag

Entry conditions (LOOSE to guarantee ≥10 trades/symbol):
- LONG: 4h_HMA_bull + RSI(7)<30 + BB_%B<0.15 + session=00-12UTC
- SHORT: 4h_HMA_bear + RSI(7)>70 + BB_%B>0.85 + session=00-12UTC
- Exit: RSI(7) crosses 50 OR stoploss hit (2.0*ATR)

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi7_bb_session_4h_trend_v1"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with %B indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Rolling mean and std
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # %B = (price - lower) / (upper - lower)
    bb_pct = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if upper[i] - lower[i] > 1e-10:
            bb_pct[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
    
    return upper, lower, bb_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_pct = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(bb_pct[i]):
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
        
        # === SESSION FILTER (00-12 UTC = London+NY overlap) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        is_session = 0 <= hour_utc <= 12
        
        # === HTF TREND BIAS (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI(7) oversold + BB at lower band + session
        if hma_4h_bull and is_session:
            if rsi_7[i] < 30.0 and bb_pct[i] < 0.15:
                desired_signal = SIZE_STRONG
            elif rsi_7[i] < 35.0 and bb_pct[i] < 0.25:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + RSI(7) overbought + BB at upper band + session
        elif hma_4h_bear and is_session:
            if rsi_7[i] > 70.0 and bb_pct[i] > 0.85:
                desired_signal = -SIZE_STRONG
            elif rsi_7[i] > 65.0 and bb_pct[i] > 0.75:
                desired_signal = -SIZE_BASE
        
        # === EXIT SIGNAL (RSI crosses 50 = mean reversion complete) ===
        if in_position and i > 100:
            if position_side > 0 and rsi_7[i] > 55.0:
                desired_signal = 0.0  # Take profit on long
            elif position_side < 0 and rsi_7[i] < 45.0:
                desired_signal = 0.0  # Take profit on short
        
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