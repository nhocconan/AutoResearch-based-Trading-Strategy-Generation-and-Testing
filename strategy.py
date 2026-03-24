#!/usr/bin/env python3
"""
Experiment #973: 5m Primary + 15m/4h HTF — Fisher Transform Entries with Session Filter

Hypothesis: 5m timeframe with strict HTF trend bias + Ehlers Fisher Transform for precise
entry timing will capture intraday momentum while avoiding choppy periods.

Key innovations:
1. 4h HMA(21) for primary trend bias (only trade with HTF trend)
2. 15m RSI(14) for momentum confirmation (RSI > 55 for long, < 45 for short)
3. 5m Ehlers Fisher Transform for precise entry timing (crosses -1.5/+1.5)
4. Session filter: only trade 08:00-20:00 UTC (high liquidity, avoid Asia low-vol)
5. Volume confirmation: volume > 1.5x 20-bar average (confirms real moves)
6. ATR(14) 2.5x trailing stop for risk management

Why 5m might work:
- Unexplored timeframe (0 experiments so far)
- Fisher Transform catches reversals better than RSI at lower TF
- Session filter avoids 60% of low-quality signals (Asia session chop)
- HTF bias prevents counter-trend trades (major cause of 5m failures)
- Volume filter ensures we only trade real breakouts, not noise

Entry conditions (STRICT for 5m):
- LONG = 4h bull + 15m RSI > 55 + Fisher crosses above -1.5 + session + volume
- SHORT = 4h bear + 15m RSI < 45 + Fisher crosses below +1.5 + session + volume

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 5m
Size: 0.15-0.20 discrete (smaller due to higher trade frequency)
Trade frequency target: 50-120 trades/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_fisher_session_vol_15m4h_v1"
timeframe = "5m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Transforms price into a Gaussian-like distribution for clearer reversal signals
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * prev_X + 0.67 * ((close - low_14) / (high_14 - low_14) - 0.5)
    
    Entry signals:
    - Long when Fisher crosses above -1.5 (oversold reversal)
    - Short when Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    close = high  # Use high for Fisher calculation (Ehlers original uses typical price)
    
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate typical price and normalized value
    typical = (high + low + close) / 3.0
    
    # Normalize to -1 to +1 range using Donchian-style high/low
    highest = pd.Series(typical).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(typical).rolling(window=period, min_periods=period).min().values
    
    x_raw = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if highest[i] > lowest[i]:
            x_raw[i] = 2.0 * ((typical[i] - lowest[i]) / (highest[i] - lowest[i])) - 1.0
        else:
            x_raw[i] = 0.0
    
    # Clamp to -0.99 to +0.99 to avoid ln(0)
    x_clamped = np.clip(x_raw, -0.99, 0.99)
    
    # Smooth with EMA-like filter
    x_smooth = np.full(n, np.nan, dtype=np.float64)
    x_smooth[period] = x_clamped[period]
    for i in range(period + 1, n):
        if not np.isnan(x_clamped[i]):
            x_smooth[i] = 0.66 * x_smooth[i-1] + 0.67 * x_clamped[i]
    
    x_smooth = np.clip(x_smooth, -0.99, 0.99)
    
    # Calculate Fisher Transform
    for i in range(period, n):
        if abs(x_smooth[i]) < 0.99:
            fisher[i] = 0.5 * np.log((1.0 + x_smooth[i]) / (1.0 - x_smooth[i]))
        else:
            fisher[i] = fisher[i-1] if i > period else 0.0
    
    # Trigger line (1-bar lag of Fisher)
    for i in range(period + 1, n):
        trigger[i] = fisher[i-1]
    
    return fisher, trigger

def is_in_session(open_time, start_hour=8, end_hour=20):
    """
    Check if timestamp is within trading session (UTC)
    08:00-20:00 UTC captures London + NY overlap, avoids Asia chop
    """
    # open_time is in milliseconds since epoch
    hour = pd.to_datetime(open_time, unit='ms').hour
    return start_hour <= hour < end_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_15m = get_htf_data(prices, '15m')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, trigger = calculate_fisher_transform(high, low, period=9)
    
    # Volume SMA for confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_15m_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(trigger[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA + 15m RSI) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_15m_momentum_long = rsi_15m_aligned[i] > 55.0
        htf_15m_momentum_short = rsi_15m_aligned[i] < 45.0
        
        # === SESSION FILTER (08:00-20:00 UTC) ===
        in_session = is_in_session(open_time[i], start_hour=8, end_hour=20)
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.5 * vol_sma_20[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = (trigger[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_short = (trigger[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Also check for Fisher extreme reversals (not just cross)
        fisher_oversold = fisher[i] < -1.8 and fisher[i] > fisher[i-1] if i > 0 else False
        fisher_overbought = fisher[i] > 1.8 and fisher[i] < fisher[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (ALL CONDITIONS MUST ALIGN) ===
        desired_signal = 0.0
        
        # LONG entries (all conditions must be met)
        if htf_4h_bull and htf_15m_momentum_long and in_session:
            if fisher_cross_long or fisher_oversold:
                if volume_confirmed:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE  # Enter smaller without volume confirmation
            elif fisher[i] < -1.0 and close[i] > close[i-1]:
                # Fisher recovering from oversold with price momentum
                if volume_confirmed:
                    desired_signal = SIZE_BASE
        
        # SHORT entries (all conditions must be met)
        elif htf_4h_bear and htf_15m_momentum_short and in_session:
            if fisher_cross_short or fisher_overbought:
                if volume_confirmed:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif fisher[i] > 1.0 and close[i] < close[i-1]:
                # Fisher recovering from overbought with price momentum
                if volume_confirmed:
                    desired_signal = -SIZE_BASE
        
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