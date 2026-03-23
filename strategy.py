#!/usr/bin/env python3
"""
Experiment #768: 30m Primary + 4h/1d HTF — RSI Pullback within HTF Trend + Session Filter

Hypothesis: After 500+ failed strategies, the pattern is clear:
1. Lower TF (30m) MUST use HTF (4h/1d) for trend direction — NOT for entries
2. 30m only for ENTRY TIMING within the HTF trend (pullback entries)
3. Session filter (8-20 UTC) captures high-liquidity periods, reduces noise
4. Relaxed volume (0.8x avg) ensures enough trades without being too strict
5. RSI thresholds must NOT be too extreme (25-35 for long, 65-75 for short)
6. ATR 2.5x stoploss is mandatory for drawdown control
7. Signal size 0.25 (conservative for lower TF fee sensitivity)

Key insight from failures (#758, #760, #765 all had 0 trades):
- Entry conditions were TOO STRICT on lower TFs
- Need to loosen RSI thresholds to generate 40-60 trades/year
- HTF trend filter prevents counter-trend trades (the real edge)

Strategy design:
1. 4h HMA(21) for primary trend bias (aligned via mtf_data)
2. 1d HMA(21) for major trend confirmation (aligned via mtf_data)
3. 30m RSI(14) for pullback entry timing (25-35 long, 65-75 short)
4. 30m ATR(14) for trailing stop (2.5x)
5. Session filter: only trade 8-20 UTC (high liquidity)
6. Volume filter: >0.8x 20-bar average (relaxed for trade frequency)
7. Discrete signals: 0.0, ±0.20, ±0.25

Target: Sharpe > 0.612, trades 40-60/year, ALL symbols positive
Timeframe: 30m (with 4h/1d HTF for direction)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_hma_4h1d_session_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster response than EMA, less lag.
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    series = pd.Series(series)
    
    # WMA helper
    def wma(s, p):
        weights = np.arange(1, p + 1)
        return s.rolling(window=p, min_periods=p).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(series, half)
    wma_full = wma(series, period)
    
    hull = wma(2 * wma_half - wma_full, sqrt_period)
    return hull.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // 3600000) % 24

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
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_30m[i]) or vol_sma_30m[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === TREND BIAS (4h HMA21 + 1d HMA21) ===
        # Both HTFs must agree for strong signal
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Strong trend: both 4h and 1d agree
        strong_bullish = trend_4h_bullish and trend_1d_bullish
        strong_bearish = trend_4h_bearish and trend_1d_bearish
        
        # Weak trend: only 4h agrees (1d neutral or opposite)
        weak_bullish = trend_4h_bullish and not trend_1d_bearish
        weak_bearish = trend_4h_bearish and not trend_1d_bullish
        
        # === VOLUME CONFIRMATION (relaxed) ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_30m[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI 25-35 (pullback in uptrend)
        rsi_long_pullback = 25 <= rsi_30m[i] <= 35
        rsi_long_extreme = rsi_30m[i] < 25
        
        # Short: RSI 65-75 (pullback in downtrend)
        rsi_short_pullback = 65 <= rsi_30m[i] <= 75
        rsi_short_extreme = rsi_30m[i] > 75
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        if in_session:
            # Strong bullish: RSI pullback 25-35
            if strong_bullish and rsi_long_pullback:
                desired_signal = BASE_SIZE if volume_confirmed else REDUCED_SIZE
            
            # Strong bullish: RSI extreme <25 (deep pullback)
            if strong_bullish and rsi_long_extreme:
                desired_signal = BASE_SIZE
            
            # Weak bullish: only enter on extreme RSI
            if weak_bullish and rsi_long_extreme and volume_confirmed:
                desired_signal = REDUCED_SIZE
            
            # Short entry in strong bearish trend
            if strong_bearish and rsi_short_pullback:
                desired_signal = -BASE_SIZE if volume_confirmed else -REDUCED_SIZE
            
            # Strong bearish: RSI extreme >75
            if strong_bearish and rsi_short_extreme:
                desired_signal = -BASE_SIZE
            
            # Weak bearish: only enter on extreme RSI
            if weak_bearish and rsi_short_extreme and volume_confirmed:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish and RSI not overbought
                if trend_4h_bullish and rsi_30m[i] < 70:
                    desired_signal = BASE_SIZE if strong_bullish else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_30m[i] > 30:
                    desired_signal = -BASE_SIZE if strong_bearish else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses bearish
            if trend_4h_bearish:
                desired_signal = 0.0
            # Exit if RSI becomes overbought
            if rsi_30m[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses bullish
            if trend_4h_bullish:
                desired_signal = 0.0
            # Exit if RSI becomes oversold
            if rsi_30m[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals