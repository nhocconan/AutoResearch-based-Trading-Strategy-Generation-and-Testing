#!/usr/bin/env python3
"""
Experiment #1573: 5m Primary + 15m/4h HTF — Session-Filtered Momentum Confluence

Hypothesis: 5m timeframe is unexplored (0 experiments). Success requires:
1. STRICT HTF trend filter (4h HMA direction) — NEVER counter-trend
2. 15m momentum confirmation (RSI 40-60 zone for continuation entries)
3. 5m entry timing (price action breakout with volume)
4. SESSION FILTER (08-20 UTC) — only trade during high liquidity hours
5. Small position size (0.15) — more trades = more fee drag

Why this should work:
- 4h HMA provides strong trend bias (avoid whipsaws)
- 15m RSI 40-60 = momentum continuation (not overbought/oversold reversals)
- 5m breakout entry = precise timing with tight stop
- Session filter = avoid low-liquidity trap moves (00-08 UTC)
- Size 0.15 = sustainable for 50-120 trades/year target

Entry logic (LOOSE enough for trades, strict enough for quality):
- LONG: 4h_HMA bullish + 15m_RSI 40-65 + 5m price > 5m_HMA8 + session active
- SHORT: 4h_HMA bearish + 15m_RSI 35-60 + 5m price < 5m_HMA8 + session active

Target: Sharpe>0.6, trades>=50 train, trades>=5 test, DD>-35%
Timeframe: 5m
Size: 0.15 discrete (small for high trade frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_momentum_4h15m_confluence_v1"
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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_volume_ma(volume, period=20):
    """Volume Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def is_session_active(open_time, start_hour=8, end_hour=20):
    """
    Check if timestamp is within trading session (08-20 UTC)
    open_time is in milliseconds since epoch
    """
    # Convert to hours UTC
    ts_seconds = open_time / 1000
    hour_utc = (ts_seconds % 86400) / 3600
    
    return start_hour <= hour_utc < end_hour

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
    
    rsi_15m_raw = calculate_rsi(df_15m['close'].values, period=14)
    rsi_15m_aligned = align_htf_to_ltf(prices, df_15m, rsi_15m_raw)
    
    # Calculate 5m indicators
    hma_8 = calculate_hma(close, period=8)
    hma_21 = calculate_hma(close, period=21)
    ema_8 = calculate_ema(close, period=8)
    ema_21 = calculate_ema(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.15  # Small size for 5m (more trades = fee drag)
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
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
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === 4H TREND BIAS (HTF direction) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === 15M MOMENTUM CONFIRMATION ===
        rsi_15m = rsi_15m_aligned[i]
        rsi_15m_bullish = 40 <= rsi_15m <= 65  # Momentum zone, not overbought
        rsi_15m_bearish = 35 <= rsi_15m <= 60  # Momentum zone, not oversold
        
        # === 5M ENTRY TRIGGERS ===
        # HMA crossover for entry timing
        hma_bullish = hma_8[i] > hma_21[i]
        hma_bearish = hma_8[i] < hma_21[i]
        
        # Price position relative to HMA
        price_above_hma8 = close[i] > hma_8[i]
        price_below_hma8 = close[i] < hma_8[i]
        
        # Volume confirmation (above average)
        volume_confirmed = volume[i] > vol_ma_20[i] * 0.8  # At least 80% of avg
        
        # 5m RSI for entry timing (not extreme)
        rsi_5m = rsi_14[i]
        rsi_5m_long_ok = 35 <= rsi_5m <= 65
        rsi_5m_short_ok = 35 <= rsi_5m <= 65
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: All conditions must align
        if (session_active and 
            price_above_4h and 
            rsi_15m_bullish and 
            hma_bullish and 
            price_above_hma8 and 
            volume_confirmed and
            rsi_5m_long_ok):
            desired_signal = SIZE
        
        # SHORT: All conditions must align
        elif (session_active and 
              price_below_4h and 
              rsi_15m_bearish and 
              hma_bearish and 
              price_below_hma8 and 
              volume_confirmed and
              rsi_5m_short_ok):
            desired_signal = -SIZE
        
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