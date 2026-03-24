#!/usr/bin/env python3
"""
Experiment #313: 5m Primary + 15m/4h HTF — Session RSI Pullback with Volume Confirmation v1

Hypothesis: 5m timeframe is unexplored but requires extreme selectivity. Using 15m/4h for 
trend direction + 5m RSI pullback entries during high-volume sessions (08-20 UTC) should 
capture intraday momentum while avoiding noise. Volume spike confirmation filters false breakouts.

Key innovations:
1. 15m HMA for immediate trend + 4h HMA for major trend - BOTH must align
2. Session filter: 08-20 UTC (London/NY overlap = highest volume, lowest noise)
3. 5m RSI pullback into trend (not breakout) - enter on retracement, not extension
4. Volume spike confirmation: entry bar volume > 1.5x 20-bar average
5. ATR stoploss: 2.0x ATR from entry price
6. Discrete sizing: 0.15 base, 0.25 when 4h aligned (conservative for 5m trade frequency)

Why this might work on 5m:
- HTF trend filter prevents counter-trend trades (biggest 5m killer)
- Session filter avoids Asian session noise and weekend gaps
- RSI pullback (not breakout) = better risk/reward on lower TF
- Volume confirmation = institutional participation signal

Target: Sharpe>0.40, DD>-40%, trades>=50 train, trades>=5 test
Position size: 0.15-0.25 (smaller due to higher trade frequency on 5m)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_5m_session_rsi_pullback_15m4h_vol_v1"
timeframe = "5m"
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

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above average"""
    n = len(volume)
    if n < period:
        return np.zeros(n, dtype=bool)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * vol_avg)
    spike[:period] = False
    return spike

def is_session_active(open_time, start_hour=8, end_hour=20):
    """
    Check if timestamp is within trading session (UTC)
    08-20 UTC captures London open through NY close
    """
    # open_time is in milliseconds since epoch
    timestamp = pd.Timestamp(open_time, unit='ms')
    hour = timestamp.hour
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
    
    # Calculate and align HTF HMA for trend bias
    hma_15m_raw = calculate_hma(df_15m['close'].values, period=21)
    hma_15m_aligned = align_htf_to_ltf(prices, df_15m, hma_15m_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (5m) indicators
    hma_5m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi_5m = calculate_rsi(close, period=14)
    rsi_5m_short = calculate_rsi(close, period=7)  # Faster RSI for pullback detection
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_5m[i]) or np.isnan(rsi_5m[i]) or np.isnan(rsi_5m_short[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        session_active = is_session_active(open_time[i], start_hour=8, end_hour=20)
        
        # === TREND ALIGNMENT (ALL TF must agree) ===
        # 5m trend
        trend_5m_bull = close[i] > hma_5m[i]
        trend_5m_bear = close[i] < hma_5m[i]
        
        # 15m trend
        trend_15m_bull = close[i] > hma_15m_aligned[i]
        trend_15m_bear = close[i] < hma_15m_aligned[i]
        
        # 4h trend (major bias)
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # All TF aligned bull
        all_bull = trend_5m_bull and trend_15m_bull and trend_4h_bull
        # All TF aligned bear
        all_bear = trend_5m_bear and trend_15m_bear and trend_4h_bear
        
        # === RSI PULLBACK DETECTION ===
        # Long: RSI dipped to 35-45 zone in uptrend (pullback, not oversold)
        rsi_pullback_long = 35.0 <= rsi_5m_short[i] <= 50.0
        # Short: RSI rallied to 55-65 zone in downtrend (pullback, not overbought)
        rsi_pullback_short = 50.0 <= rsi_5m_short[i] <= 65.0
        
        # RSI confirmation (main RSI not extreme)
        rsi_ok_long = 40.0 <= rsi_5m[i] <= 60.0
        rsi_ok_short = 40.0 <= rsi_5m[i] <= 60.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_spike[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Only trade during active session
        if session_active:
            # LONG: All bull + RSI pullback + volume spike
            if all_bull and rsi_pullback_long and rsi_ok_long and vol_confirmed:
                desired_signal = SIZE_STRONG
            
            # SHORT: All bear + RSI pullback + volume spike
            elif all_bear and rsi_pullback_short and rsi_ok_short and vol_confirmed:
                desired_signal = -SIZE_STRONG
            
            # Weaker signal: 15m+5m aligned but 4h neutral/opposite
            elif trend_5m_bull and trend_15m_bull and rsi_pullback_long and vol_confirmed:
                desired_signal = SIZE_BASE
            elif trend_5m_bear and trend_15m_bear and rsi_pullback_short and vol_confirmed:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR from entry) ===
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
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals