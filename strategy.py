#!/usr/bin/env python3
"""
Experiment #953: 5m Primary + 15m/4h HTF — Session-Filtered Trend Following

Hypothesis: 5m timeframe is extremely noisy and requires strict HTF filtering.
Most 5m strategies fail due to fee drag from overtrading. This strategy uses:
1. 4h HMA(21) for PRIMARY trend direction (never trade counter-trend)
2. 15m RSI(14) for momentum confirmation (RSI>55 for long, <45 for short)
3. 5m EMA(21) pullback entries in trend direction
4. Session filter: ONLY 08:00-20:00 UTC (London/NY overlap = highest volume)
5. Volume ratio filter: taker_buy_volume/volume > 0.55 for longs
6. ATR(14) 2.0x trailing stoploss
7. Small position size (0.15-0.20) due to higher trade frequency

Why this might work on 5m:
- 4h trend filter eliminates 50% of false signals
- Session filter avoids low-volume Asian session whipsaws
- RSI momentum filter ensures we enter with momentum, not against it
- Pullback to EMA(21) gives better entry than chasing breakouts
- Small size (0.15) reduces fee drag impact

Target: Sharpe>0.5, trades>=50 train, trades>=5 test, DD>-30%
Timeframe: 5m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_hma_rsi_pullback_15m4h_v1"
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
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def get_session_mask(prices):
    """
    Session filter: 08:00-20:00 UTC only (London/NY overlap)
    Returns boolean mask where True = trade allowed
    """
    n = len(prices)
    mask = np.zeros(n, dtype=bool)
    
    # Parse open_time to get hour
    for i in range(n):
        open_time = prices['open_time'].iloc[i]
        # open_time is in milliseconds since epoch
        hour = (open_time // 3600000) % 24
        # Trade only during 08:00-20:00 UTC
        if 8 <= hour < 20:
            mask[i] = True
    
    return mask

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
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
    ema_5m_21 = calculate_ema(close, period=21)
    ema_5m_50 = calculate_ema(close, period=50)
    rsi_5m_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (buying pressure)
    volume_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if volume[i] > 1e-10:
            volume_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            volume_ratio[i] = 0.5
    
    # Session mask
    session_mask = get_session_mask(prices)
    
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
        
        if np.isnan(ema_5m_21[i]) or np.isnan(ema_5m_50[i]):
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
        
        # === SESSION FILTER (MANDATORY for 5m) ===
        if not session_mask[i]:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m MOMENTUM (RSI) ===
        rsi_15m = rsi_15m_aligned[i]
        momentum_bull = rsi_15m > 55.0
        momentum_bear = rsi_15m < 45.0
        
        # === 5m PULLBACK ENTRY ===
        # Long: price pulls back to EMA21 but EMA21 > EMA50 (trend up)
        pullback_long = (close[i] <= ema_5m_21[i] * 1.002) and (ema_5m_21[i] > ema_5m_50[i])
        # Short: price rallies to EMA21 but EMA21 < EMA50 (trend down)
        pullback_short = (close[i] >= ema_5m_21[i] * 0.998) and (ema_5m_21[i] < ema_5m_50[i])
        
        # === VOLUME CONFIRMATION ===
        volume_buy_pressure = volume_ratio[i] > 0.55
        volume_sell_pressure = volume_ratio[i] < 0.45
        
        # === 5m RSI CONFIRMATION (avoid overbought/oversold entries) ===
        rsi_5m = rsi_5m_14[i]
        rsi_long_ok = 35.0 < rsi_5m < 70.0
        rsi_short_ok = 30.0 < rsi_5m < 65.0
        
        # === ENTRY LOGIC (ALL CONDITIONS MUST ALIGN) ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m momentum bull + 5m pullback + volume + RSI ok
        if htf_4h_bull and momentum_bull and pullback_long and volume_buy_pressure and rsi_long_ok:
            desired_signal = SIZE_STRONG
        # LONG (weaker): same but without volume filter
        elif htf_4h_bull and momentum_bull and pullback_long and rsi_long_ok:
            desired_signal = SIZE_BASE
        
        # SHORT: 4h bear + 15m momentum bear + 5m pullback + volume + RSI ok
        elif htf_4h_bear and momentum_bear and pullback_short and volume_sell_pressure and rsi_short_ok:
            desired_signal = -SIZE_STRONG
        # SHORT (weaker): same but without volume filter
        elif htf_4h_bear and momentum_bear and pullback_short and rsi_short_ok:
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