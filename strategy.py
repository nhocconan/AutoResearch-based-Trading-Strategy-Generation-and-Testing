#!/usr/bin/env python3
"""
Experiment #1301: 15m Primary + 1h/4h HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m strategies have failed due to either (1) too many trades with fee drag,
or (2) over-filtering resulting in 0 trades. This strategy finds the middle ground:

1. 15m HMA(21) for fast trend detection (responsive to intraday moves)
2. 1h HMA(21) for immediate trend confirmation (not too slow like 4h/12h)
3. 4h HMA(21) for regime bias (only trade with major trend)
4. RSI(7) pullback entries (30-50 for long in uptrend, 50-70 for short in downtrend)
5. Session filter: 00-14 UTC only (London+NY overlap = best liquidity)
6. ATR(14) 2.0x trailing stop for risk management

Key differences from failed #1297:
- Use 1h instead of 12h for HTF (12h too slow for 15m entries)
- RSI pullback (30-70) instead of extremes (20-80 too rare)
- Session filter reduces trades but improves win rate
- Smaller position size (0.15-0.20) for 15m frequency

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_session_1h4h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close = np.asarray(close, dtype=np.float64)
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
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
    """Average True Range using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, lookback=7):
    """RSI using Wilder's smoothing method"""
    n = len(close)
    if n < lookback + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.full(n, np.nan, dtype=np.float64)
    avg_loss = np.full(n, np.nan, dtype=np.float64)
    
    avg_gain[lookback] = np.mean(gains[1:lookback+1])
    avg_loss[lookback] = np.mean(losses[1:lookback+1])
    
    for i in range(lookback + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (lookback - 1) + gains[i]) / lookback
        avg_loss[i] = (avg_loss[i-1] * (lookback - 1) + losses[i]) / lookback
    
    rs = np.zeros(n, dtype=np.float64)
    rsi = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs[i]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, lookback=7)
    
    # Volume MA for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
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
        
        # === SESSION FILTER (00-14 UTC = London+NY overlap) ===
        try:
            timestamp = prices.iloc[i]['open_time']
            if isinstance(timestamp, (int, np.integer)):
                hour = (timestamp // 3600000) % 24
            else:
                hour = pd.to_datetime(timestamp).hour
            in_session = 0 <= hour <= 14
        except:
            in_session = True  # Default to trading if can't parse
        
        # === TREND DIRECTION ===
        # 1h HMA slope (compare to 3 bars ago for stability)
        hma_1h_slope = 0.0
        if i >= 3 and not np.isnan(hma_1h_aligned[i-3]):
            hma_1h_slope = hma_1h_aligned[i] - hma_1h_aligned[i-3]
        
        # 4h HMA slope for regime
        hma_4h_slope = 0.0
        if i >= 3 and not np.isnan(hma_4h_aligned[i-3]):
            hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-3]
        
        # Price position relative to HMAs
        price_above_1h = close[i] > hma_1h_aligned[i]
        price_below_1h = close[i] < hma_1h_aligned[i]
        
        # === MOMENTUM (RSI pullback) ===
        rsi = rsi_7[i]
        
        # Volume confirmation (optional, loose threshold)
        vol_confirm = True
        if not np.isnan(vol_ma20[i]) and vol_ma20[i] > 0:
            vol_confirm = volume[i] > 0.5 * vol_ma20[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: 1h trend rising + 4h neutral/rising + RSI pullback (30-50)
        if in_session and hma_1h_slope > 0 and hma_4h_slope >= -0.001:
            if price_above_1h and vol_confirm:
                if 30 <= rsi <= 55:
                    if rsi <= 42:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
        
        # SHORT: 1h trend falling + 4h neutral/falling + RSI bounce (45-70)
        elif in_session and hma_1h_slope < 0 and hma_4h_slope <= 0.001:
            if price_below_1h and vol_confirm:
                if 45 <= rsi <= 70:
                    if rsi >= 58:
                        desired_signal = -SIZE_STRONG
                    else:
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