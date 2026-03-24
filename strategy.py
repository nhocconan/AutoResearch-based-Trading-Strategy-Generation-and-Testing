#!/usr/bin/env python3
"""
Experiment #853: 5m Primary + 15m/4h HTF — Fisher Transform + Session Filter

Hypothesis: 5m timeframe with strict HTF trend filter and session constraints
can capture intraday momentum while avoiding fee drag. Fisher Transform provides
superior reversal detection vs RSI on lower TF. 4h HMA for primary trend, 15m
RSI for momentum confirmation, 5m Fisher for precise entry timing.

Key innovations:
1. 4h HMA(21) for primary trend bias (never trade counter-trend)
2. 15m RSI(7) for momentum confirmation (RSI>55 long, RSI<45 short)
3. 5m Fisher Transform(9) for entry timing (crosses -1.5 long, +1.5 short)
4. Session filter: 08-20 UTC only (London/NY overlap = highest liquidity)
5. Volume spike filter: volume > 1.5x 20-bar MA (avoid low-liquidity traps)
6. ATR(14) 2.0x trailing stop for risk management
7. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller due to higher trade frequency)

Why this might work on 5m:
- Fisher Transform catches reversals faster than RSI (proven on crypto)
- Session filter eliminates Asian session chop (00-08 UTC)
- Volume filter avoids false breakouts on low liquidity
- HTF trend filter prevents counter-trend trades (major failure mode)

Target: Sharpe>0.45, trades>=50 train, trades>=5 test, DD>-40%, trades/year<120
Timeframe: 5m
Size: 0.15-0.20 discrete (smaller than higher TF due to fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_fisher_session_volume_4h15m_v1"
timeframe = "5m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Fisher = 0.5 * ln((1 + value) / (1 - value))
    value = 0.66 * prev_value + 0.34 * (2 * (close - low_5m) / (high_5m - low_5m) - 1)
    
    Catches reversals faster than RSI. Long when Fisher crosses above -1.5,
    short when crosses below +1.5.
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Use high/low for Fisher calculation (more accurate than close only)
    # We'll approximate with close since we don't have separate HL arrays here
    # Actually we do have high/low passed separately, but for simplicity use close range
    
    fisher = np.full(n, np.nan)
    value = np.zeros(n)
    
    # Calculate median price for Fisher
    median = (close)  # Using close as approximation
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(close[i - period + 1:i + 1])
        lowest = np.min(close[i - period + 1:i + 1])
        
        if highest - lowest < 1e-10:
            value[i] = value[i-1] if i > 0 else 0.0
        else:
            raw_value = 0.66 * ((close[i] - lowest) / (highest - lowest) * 2 - 1) + \
                       0.34 * (value[i-1] if i > 0 else 0.0)
            # Clamp to prevent division by zero
            raw_value = np.clip(raw_value, -0.999, 0.999)
            value[i] = raw_value
        
        # Fisher transform
        if abs(value[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i]))
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_session_filter(open_time):
    """
    Session filter: 08-20 UTC only (London/NY overlap)
    open_time is in milliseconds since epoch
    Returns boolean array: True if within session
    """
    n = len(open_time)
    session_mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        # Convert ms to hours UTC
        hour_utc = (open_time[i] // 3600000) % 24
        # 08-20 UTC = London open through NY close
        if 8 <= hour_utc < 20:
            session_mask[i] = True
    
    return session_mask

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_15m = get_htf_data(prices, '15m')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=7)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    fisher_5m = calculate_fisher(close, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    session_mask = calculate_session_filter(open_time)
    
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
    
    # Fisher crossover tracking
    prev_fisher = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]) or np.isnan(fisher_5m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ma20[i]) or vol_ma20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        if not session_mask[i]:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher_5m[i]
            continue
        
        # === VOLUME FILTER ===
        volume_spike = volume[i] > 1.5 * vol_ma20[i]
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m MOMENTUM (RSI) ===
        rsi_15m_bull = rsi_15m_aligned[i] > 55.0
        rsi_15m_bear = rsi_15m_aligned[i] < 45.0
        
        # === 5m FISHER TRANSFORM ENTRY ===
        fisher_long_signal = False
        fisher_short_signal = False
        
        if not np.isnan(prev_fisher):
            # Long: Fisher crosses above -1.5 from below
            fisher_long_signal = (prev_fisher < -1.5) and (fisher_5m[i] >= -1.5)
            # Short: Fisher crosses below +1.5 from above
            fisher_short_signal = (prev_fisher > 1.5) and (fisher_5m[i] <= 1.5)
        
        prev_fisher = fisher_5m[i]
        
        # === ENTRY LOGIC (ALL CONDITIONS MUST ALIGN) ===
        desired_signal = 0.0
        
        if htf_4h_bull and rsi_15m_bull:
            # Bullish alignment: only long entries
            if fisher_long_signal and volume_spike:
                desired_signal = SIZE_STRONG
            elif fisher_5m[i] < -1.0 and volume_spike:
                # Deep oversold in uptrend
                desired_signal = SIZE_BASE
        
        elif htf_4h_bear and rsi_15m_bear:
            # Bearish alignment: only short entries
            if fisher_short_signal and volume_spike:
                desired_signal = -SIZE_STRONG
            elif fisher_5m[i] > 1.0 and volume_spike:
                # Deep overbought in downtrend
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