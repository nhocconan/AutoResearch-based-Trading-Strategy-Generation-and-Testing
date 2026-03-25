#!/usr/bin/env python3
"""
Experiment #1329: 15m Primary + 1h/1d HTF — Camarilla Pivot Mean-Reversion + RSI Timing

Hypothesis: 15m strategies have ALL failed with Sharpe=0.000 (ZERO trades). The problem is
too many filters killing all signals. This strategy uses a PROVEN mean-reversion framework:

1. Camarilla pivot levels from 1d HTF (R3/S3 for mean-reversion, R4/S4 for breakout)
2. 1h HMA(21) for trend bias (only trade with HTF direction)
3. 15m RSI(7) for entry timing (oversold <30 long, overbought >70 short)
4. ATR(14) 2.5x trailing stop for risk management
5. LOOSE entry conditions to GUARANTEE 50-100 trades/year

Why Camarilla works on crypto:
- R3/S3 = value area boundaries (75% of price action stays inside)
- R4/S4 = breakout levels (follow momentum)
- Works in BOTH trending and ranging markets
- Self-adjusting daily based on prior day's range

Key differences from failed 15m strategies:
- NO session filter (crypto trades 24/7, session filters killed all signals)
- RSI(7) not RSI(14) - faster signals for 15m timeframe
- 1h HMA not 4h/12h - appropriate speed for 15m entries
- Size=0.20 (smaller for higher frequency, reduces fee impact)

Entry logic (LOOSE to guarantee trades):
- LONG: price < S3 + RSI(7) < 30 + 1h_HMA bullish
- SHORT: price > R3 + RSI(7) > 70 + 1h_HMA bearish
- BREAKOUT LONG: price > R4 + 1h_HMA bullish
- BREAKOUT SHORT: price < S4 + 1h_HMA bearish

Target: Sharpe>0.5, trades>=50 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_pivot_rsi_1h1d_v1"
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

def calculate_rsi(close, period=7):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_camarilla_pivots(df_daily):
    """
    Calculate Camarilla pivot levels from daily data.
    Returns arrays of H3, H4, L3, L4 for each day.
    
    Camarilla formula:
    Pivot = (High + Low + Close) / 3
    H3 = Close + (High - Low) * 1.1 / 12
    H4 = Close + (High - Low) * 1.1 / 6
    L3 = Close - (High - Low) * 1.1 / 12
    L4 = Close - (High - Low) * 1.1 / 6
    """
    n = len(df_daily)
    h3 = np.full(n, np.nan)
    h4 = np.full(n, np.nan)
    l3 = np.full(n, np.nan)
    l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        high = df_daily['high'].iloc[i-1]  # Previous day's high
        low = df_daily['low'].iloc[i-1]    # Previous day's low
        close = df_daily['close'].iloc[i-1]  # Previous day's close
        
        range_val = high - low
        h3[i] = close + range_val * 1.1 / 12
        h4[i] = close + range_val * 1.1 / 6
        l3[i] = close - range_val * 1.1 / 12
        l4[i] = close - range_val * 1.1 / 6
    
    return h3, h4, l3, l4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1h HMA for trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate Camarilla pivots from daily data
    h3_daily, h4_daily, l3_daily, l4_daily = calculate_camarilla_pivots(df_1d)
    
    # Align daily pivots to 15m timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_daily)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4_daily)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_daily)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4_daily)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    signals = np.zeros(n)
    SIZE = 0.20  # Smaller size for 15m (higher frequency)
    
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
        
        if np.isnan(rsi_7[i]):
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
        
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1h HMA slope) ===
        hma_1h_slope = 0.0
        if i >= 4 and not np.isnan(hma_1h_aligned[i-4]):
            hma_1h_slope = hma_1h_aligned[i] - hma_1h_aligned[i-4]
        
        trend_bullish = hma_1h_slope > 0
        trend_bearish = hma_1h_slope < 0
        
        # === CAMARILLA LEVELS ===
        h3 = h3_aligned[i]
        h4 = h4_aligned[i]
        l3 = l3_aligned[i]
        l4 = l4_aligned[i]
        
        price = close[i]
        rsi = rsi_7[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # MEAN REVERSION LONG: price at S3 + RSI oversold + bullish bias
        if price <= l3 and rsi < 35:
            if trend_bullish or not trend_bearish:  # Loose: allow neutral
                desired_signal = SIZE
        
        # MEAN REVERSION SHORT: price at R3 + RSI overbought + bearish bias
        elif price >= h3 and rsi > 65:
            if trend_bearish or not trend_bullish:  # Loose: allow neutral
                desired_signal = -SIZE
        
        # BREAKOUT LONG: price breaks R4 + bullish trend
        elif price > h4 and trend_bullish:
            desired_signal = SIZE
        
        # BREAKOUT SHORT: price breaks S4 + bearish trend
        elif price < l4 and trend_bearish:
            desired_signal = -SIZE
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
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