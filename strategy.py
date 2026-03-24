#!/usr/bin/env python3
"""
Experiment #461: 15m Primary + 1h/4h HTF — Session-Filtered Mean Reversion

Hypothesis: 15m timeframe is ideal for intraday mean reversion with HTF trend filter.
Key insights from 200+ failed experiments:
- 15m has ZERO successful experiments — this is unexplored territory
- Most 15m failures: either 0 trades (too strict) or >300 trades/year (fee drag)
- Session filter CRITICAL: crypto volume peaks 00-12 UTC (London+NY overlap)
- HTF should be 1h (not 4h/1d) for 15m entries — 4h too slow for intraday

Strategy Design:
1. HTF TREND: 1h HMA(21) for direction bias (faster than 4h for 15m entries)
2. MEAN REVERSION: 15m RSI(7) extremes (<25/>75) + BB touch for entries
3. SESSION FILTER: Only trade 00-12 UTC (high volume, avoid Asia chop)
4. REGIME FILTER: ADX < 25 = mean revert, ADX > 25 = trend follow
5. POSITION SIZE: 0.15-0.20 (smaller for 15m frequency vs 6h's 0.25-0.30)
6. STOPLOSS: 2.5x ATR from entry (wider for 15m noise)

Target: Sharpe>0.45, DD>-35%, trades>=150 train (40/year), trades>=15 test
Timeframe: 15m (FIRST 15m experiment with proper session + HTF filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_session_rsi7_1h_trend_bb_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    dx[:] = np.nan
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, sma

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000.0
    hour = (ts_seconds % 86400) / 3600.0
    return int(hour)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF HMA for trend bias
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    rsi_fast = calculate_rsi(close, period=7)  # Faster RSI for 15m
    rsi_slow = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_dev=2.0)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_fast[i]) or np.isnan(adx[i]):
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
        
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        hour = get_session_hour(open_time[i])
        is_session_active = (hour >= 0 and hour < 12)
        
        # === REGIME DETECTION ===
        # ADX < 25 = mean reversion regime
        # ADX >= 25 = trend regime
        is_mean_revert = adx[i] < 25.0
        is_trend = adx[i] >= 25.0
        
        # === HTF TREND BIAS (1h + 4h agreement) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # Both HTF agree = strong bias
        htf_strong_bull = htf_1h_bull and htf_4h_bull
        htf_strong_bear = htf_1h_bear and htf_4h_bear
        htf_mixed = not htf_strong_bull and not htf_strong_bear
        
        # === 15m TREND ===
        price_above_hma = close[i] > hma_15m[i]
        price_below_hma = close[i] < hma_15m[i]
        price_above_sma50 = close[i] > sma_50[i]
        price_above_sma200 = close[i] > sma_200[i]
        
        # === RSI EXTREMES (FASTER: 7-period) ===
        rsi_oversold = rsi_fast[i] < 25.0
        rsi_overbought = rsi_fast[i] > 75.0
        rsi_extreme_oversold = rsi_fast[i] < 15.0
        rsi_extreme_overbought = rsi_fast[i] > 85.0
        
        # === BB TOUCH ===
        touch_lower = close[i] <= bb_lower[i] if not np.isnan(bb_lower[i]) else False
        touch_upper = close[i] >= bb_upper[i] if not np.isnan(bb_upper[i]) else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only trade during active session
        if is_session_active:
            # REGIME 1: MEAN REVERSION (ADX < 25)
            if is_mean_revert:
                # Long: RSI extreme oversold + BB lower touch + HTF not strongly bear
                if rsi_oversold and touch_lower and not htf_strong_bear:
                    if price_above_sma200:  # Long-term bullish bias
                        desired_signal = SIZE_STRONG
                    elif htf_mixed or htf_strong_bull:
                        desired_signal = SIZE_BASE
                
                # Short: RSI extreme overbought + BB upper touch + HTF not strongly bull
                elif rsi_overbought and touch_upper and not htf_strong_bull:
                    if not price_above_sma200:  # Long-term bearish bias
                        desired_signal = -SIZE_STRONG
                    elif htf_mixed or htf_strong_bear:
                        desired_signal = -SIZE_BASE
                
                # Extra: RSI divergence entry (RSI(7) < RSI(14) while oversold)
                elif rsi_extreme_oversold and rsi_fast[i] < rsi_slow[i]:
                    if htf_strong_bull or htf_mixed:
                        desired_signal = SIZE_BASE
                
                elif rsi_extreme_overbought and rsi_fast[i] > rsi_slow[i]:
                    if htf_strong_bear or htf_mixed:
                        desired_signal = -SIZE_BASE
            
            # REGIME 2: TREND FOLLOWING (ADX >= 25)
            elif is_trend:
                # Long: HTF strong bull + 15m pullback to HMA + RSI not overbought
                if htf_strong_bull and price_below_hma and rsi_fast[i] < 60.0:
                    if rsi_fast[i] > 30.0:  # Not extreme oversold (avoid catching falling knife)
                        desired_signal = SIZE_BASE
                
                # Short: HTF strong bear + 15m rally to HMA + RSI not oversold
                elif htf_strong_bear and price_above_hma and rsi_fast[i] > 40.0:
                    if rsi_fast[i] < 70.0:  # Not extreme overbought
                        desired_signal = -SIZE_BASE
                
                # Breakout: Price breaks BB with HTF confirmation
                elif htf_strong_bull and close[i] > bb_upper[i] and rsi_fast[i] < 70.0:
                    desired_signal = SIZE_BASE
                
                elif htf_strong_bear and close[i] < bb_lower[i] and rsi_fast[i] > 30.0:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss (2.5x ATR for 15m noise)
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
        
        signals[i] = final_signal
    
    return signals