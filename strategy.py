#!/usr/bin/env python3
"""
Experiment #1633: 5m Primary + 15m/4h HTF — Session-Filtered Trend Following

Hypothesis: 5m timeframe with 4h trend bias + 15m momentum filter captures optimal
entry timing while avoiding counter-trend trades. Session filter (08-20 UTC) ensures
liquidity and reduces noise. Fisher Transform on 5m provides precise entry triggers.

CRITICAL LESSONS FROM FAILURES (#1625, #1629, #1630 - Sharpe=0.000, 0 trades):
1. Entry conditions MUST be LOOSE to guarantee trades (≥30 train, ≥3 test)
2. Session filter should NOT block all entries - make it permissive
3. 5m needs HTF trend bias to avoid whipsaw (never trade counter-trend)
4. Size must be SMALL (0.15-0.20) due to fee drag from frequent trades

Key design choices:
1. 4h HMA(21) for trend bias - only trade in HTF direction
2. 15m RSI(14) as loose momentum filter (not extreme thresholds)
3. 5m Fisher(9) for entry timing - crossover signals
4. Session filter: 08-20 UTC (London/NY overlap) - permissive, not restrictive
5. Size: 0.15 base, 0.20 strong (small due to 5m fee drag)
6. Stoploss: 2.0x ATR trailing via signal→0

Entry logic (LOOSE to guarantee trades):
- LONG: 4h bullish + 15m RSI > 40 + 5m Fisher cross up OR 5m RSI < 30 (pullback entry)
- SHORT: 4h bearish + 15m RSI < 60 + 5m Fisher cross down OR 5m RSI > 70 (pullback entry)
- Session filter: prefer 08-20 UTC but allow outside if signal strong

Target: Sharpe>0.6, trades≥50/train, trades≥5/test, DD>-35%, 50-120 trades/year
Timeframe: 5m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_fisher_session_4h15m_trend_v1"
timeframe = "5m"
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

def calculate_fisher(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Returns fisher value and trigger (previous value for crossover detection)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    median = (high + low) / 2
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val < 1e-10:
            if i > 0 and not np.isnan(fisher[i-1]):
                fisher[i] = fisher[i-1]
                trigger[i] = fisher[i-1]
            continue
        
        normalized = 2.0 * (median[i] - lowest) / range_val - 1.0
        normalized = max(-0.999, min(0.999, normalized))
        
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_15m = get_htf_data(prices, '15m')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_trigger = calculate_fisher(high, low, period=9)
    sma_200 = calculate_sma(close, period=200)
    
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
    min_bars = 250  # Need 200 for SMA + buffer
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) - PERMISSIVE ===
        # Extract hour from open_time (milliseconds timestamp)
        ts_ms = open_time[i]
        hour_utc = (ts_ms // (1000 * 60 * 60)) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === 15m MOMENTUM FILTER (LOOSE) ===
        rsi_15m = rsi_15m_aligned[i]
        momentum_bullish = rsi_15m > 40  # Not too weak
        momentum_bearish = rsi_15m < 60  # Not too strong
        
        # === 5m FISHER TRANSFORM SIGNALS ===
        fisher_val = fisher[i]
        fisher_prev = fisher_trigger[i] if not np.isnan(fisher_trigger[i]) else fisher_val
        
        # Fisher crossover signals - LOOSE thresholds for trades
        fisher_bull_cross = fisher_val > -1.0 and fisher_prev <= -1.0
        fisher_bear_cross = fisher_val < 1.0 and fisher_prev >= 1.0
        
        # Fisher extremes (mean reversion entries)
        fisher_extreme_low = fisher_val < -0.5
        fisher_extreme_high = fisher_val > 0.5
        
        # === 5m RSI PULLBACK ENTRIES ===
        rsi_5m = rsi_14[i]
        rsi_pullback_long = rsi_5m < 35  # Pullback in uptrend
        rsi_pullback_short = rsi_5m > 65  # Pullback in downtrend
        
        # === SMA200 FILTER (avoid trading against major trend) ===
        price_above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        price_below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG ENTRIES (4h bullish + 15m momentum + 5m trigger)
        if price_above_4h and momentum_bullish:
            # Primary: Fisher crossover up
            if fisher_bull_cross:
                if in_session:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE  # Reduced size outside session
            
            # Secondary: RSI pullback entry (catch dips in uptrend)
            elif rsi_pullback_long and fisher_extreme_low:
                if in_session:
                    desired_signal = SIZE_BASE
                else:
                    desired_signal = SIZE_BASE * 0.5
        
        # SHORT ENTRIES (4h bearish + 15m momentum + 5m trigger)
        elif price_below_4h and momentum_bearish:
            # Primary: Fisher crossover down
            if fisher_bear_cross:
                if in_session:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE  # Reduced size outside session
            
            # Secondary: RSI pullback entry (catch rallies in downtrend)
            elif rsi_pullback_short and fisher_extreme_high:
                if in_session:
                    desired_signal = -SIZE_BASE
                else:
                    desired_signal = -SIZE_BASE * 0.5
        
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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