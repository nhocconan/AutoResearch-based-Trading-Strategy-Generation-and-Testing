#!/usr/bin/env python3
"""
Experiment #1585: 15m Primary + 4h/1d HTF — Daily CPR + RSI Mean Reversion

Hypothesis: 15m timeframe is UNDEREXPLORED (0 successful experiments). Using Daily
Central Pivot Range (CPR) as key S/R levels combined with 4h HMA trend bias and
15m RSI(7) for entry timing should capture intraday mean-reversion with HTF confirmation.

Key innovations vs failed 15m attempts (#1577, #1581):
1. DAILY CPR (Central Pivot Range): BC/TC from 1d HTF - proven pivot levels that
   act as magnet zones. Long when price > TC + narrow CPR, short when < BC.
2. 4h HMA(21) trend filter: Only trade in direction of 4h trend (prevents counter-trend)
3. RSI(7) LOOSE thresholds: 25/75 (not 20/80) to guarantee trades
4. Session filter: 00-12 UTC preferred (London/NY overlap = better liquidity)
5. ATR stoploss: 2.5x ATR trailing stop on all positions

Why this should beat failed 15m strategies:
- Failed #1577/#1581 had too many confluence filters = 0 trades
- This uses LOOSE RSI(25/75) + simple CPR levels = frequent entries
- 4h trend filter prevents major counter-trend losses
- Position size 0.15-0.20 (smaller for 15m frequency)

Entry logic (LOOSE to guarantee ≥30 trades/train):
- LONG: 4h_HMA bullish + price > 1d_TC + RSI(7) < 35 + (optional session)
- SHORT: 4h_HMA bearish + price < 1d_BC + RSI(7) > 65 + (optional session)
- CPR narrow (< 1% of price) = breakout day, wider = range day

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_rsi_session_4h1d_v1"
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
    """Relative Strength Index - shorter period for 15m"""
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

def calculate_cpr_from_1d(df_1d):
    """
    Calculate Daily Central Pivot Range (CPR) from 1d data
    CPR consists of: BC (Bottom Central), TC (Top Central), Pivot
    Narrow CPR = potential breakout day, Wide CPR = range day
    """
    n = len(df_1d)
    if n < 2:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    pivot = np.full(n, np.nan, dtype=np.float64)
    bc = np.full(n, np.nan, dtype=np.float64)
    tc = np.full(n, np.nan, dtype=np.float64)
    cpr_width = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        
        # Standard pivot calculation
        pivot[i] = (prev_high + prev_low + prev_close) / 3.0
        bc[i] = (prev_high + prev_low) / 2.0
        tc[i] = (pivot[i] - bc[i]) + pivot[i]
        
        # CPR width as % of price (for narrow/wide detection)
        if pivot[i] > 0:
            cpr_width[i] = abs(tc[i] - bc[i]) / pivot[i] * 100.0
    
    return pivot, bc, tc, cpr_width

def calculate_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # Convert ms to seconds, then to datetime
    timestamps = pd.to_datetime(open_time, unit='ms', utc=True)
    return timestamps.hour.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d CPR levels
    pivot_1d_raw, bc_1d_raw, tc_1d_raw, cpr_width_1d_raw = calculate_cpr_from_1d(df_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    cpr_width_1d_aligned = align_htf_to_ltf(prices, df_1d, cpr_width_1d_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    session_hour = calculate_session_hour(open_time)
    
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
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bc_1d_aligned[i]) or np.isnan(tc_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === CPR LEVELS ===
        current_bc = bc_1d_aligned[i]
        current_tc = tc_1d_aligned[i]
        current_pivot = pivot_1d_aligned[i]
        cpr_width = cpr_width_1d_aligned[i] if not np.isnan(cpr_width_1d_aligned[i]) else 1.0
        
        # Narrow CPR = breakout potential (< 0.5% of price)
        is_narrow_cpr = cpr_width < 0.5
        
        # === RSI SIGNALS (LOOSE thresholds for trades) ===
        rsi_val = rsi_7[i]
        rsi_oversold = rsi_val < 35  # Loose: not 30
        rsi_overbought = rsi_val > 65  # Loose: not 70
        rsi_extreme_low = rsi_val < 25
        rsi_extreme_high = rsi_val > 75
        
        # === SESSION FILTER (prefer 00-12 UTC) ===
        hour = session_hour[i]
        is_prime_session = 0 <= hour <= 12  # London/NY overlap
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + price > TC (above CPR) + RSI oversold bounce
        # OR: 4h bullish + price near pivot + RSI extreme low
        if price_above_4h:
            if close[i] > current_tc and rsi_oversold:
                # Above CPR, RSI oversold = pullback entry
                desired_signal = SIZE_STRONG if is_prime_session else SIZE_BASE
            elif abs(close[i] - current_pivot) / current_pivot < 0.01 and rsi_extreme_low:
                # Near pivot, extreme RSI = strong mean reversion
                desired_signal = SIZE_STRONG
        
        # SHORT: 4h bearish + price < BC (below CPR) + RSI overbought
        # OR: 4h bearish + price near pivot + RSI extreme high
        elif price_below_4h:
            if close[i] < current_bc and rsi_overbought:
                # Below CPR, RSI overbought = pullback entry
                desired_signal = -SIZE_STRONG if is_prime_session else -SIZE_BASE
            elif abs(close[i] - current_pivot) / current_pivot < 0.01 and rsi_extreme_high:
                # Near pivot, extreme RSI = strong mean reversion
                desired_signal = -SIZE_STRONG
        
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