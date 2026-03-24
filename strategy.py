#!/usr/bin/env python3
"""
Experiment #296: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume + Session v1

Hypothesis: Simpler is better. Complex regime detection (Choppiness, Funding) has failed repeatedly.
Return to proven pattern: HTF trend direction + LTF pullback entries + volume confirmation.

Key changes from failed experiments:
1. REMOVED Choppiness Index — adds complexity without edge (failed in #284, #286, #287)
2. REMOVED Funding Rate — data alignment issues, inconsistent results (#284, #287, #290)
3. REMOVED Connors RSI — too many parameters, overfitting risk
4. SIMPLIFIED to: 4h HMA trend + 1d HMA major trend + 30m RSI pullback + volume filter
5. SESSION FILTER: Only trade 08-20 UTC (high liquidity, less whipsaw)
6. VOLUME CONFIRMATION: Taker buy ratio > 0.55 for longs, < 0.45 for shorts

Entry Logic:
- Long: 4h HMA bull + 1d HMA bull + RSI(14) pullback to 40-55 + volume confirmation + session
- Short: 4h HMA bear + 1d HMA bear + RSI(14) pullback to 45-60 + volume confirmation + session

Position sizing: 0.25 base, 0.30 when both 4h+1d aligned (discrete levels)
Stoploss: 2.5x ATR from entry price
Session: 08-20 UTC only (reduces overnight gap risk)

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test, 40-80 trades/year
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_volume_session_4h1d_v1"
timeframe = "30m"
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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column"""
    # open_time is in milliseconds since epoch
    open_time_ms = prices['open_time'].values
    # Convert to seconds, then to datetime
    open_time_s = open_time_ms / 1000.0
    # Extract hour (UTC)
    hours = ((open_time_s % 86400) / 3600).astype(int)
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Extract UTC hours for session filter
    hours = get_hour_from_open_time(prices)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    hma_30m = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Volume ratio (taker buy / total volume)
    taker_ratio = np.zeros(n)
    taker_ratio[:] = np.nan
    for i in range(n):
        if volume[i] > 1e-10:
            taker_ratio[i] = taker_buy_volume[i] / volume[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # Trade cooldown to limit frequency
    last_trade_bar = -100
    cooldown_bars = 20  # Minimum 10 hours between trades on 30m
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_30m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200[i]) or np.isnan(taker_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === HTF TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 30m HMA TREND ===
        hma_30m_bull = close[i] > hma_30m[i]
        hma_30m_bear = close[i] < hma_30m[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME CONFIRMATION ===
        volume_bull = taker_ratio[i] > 0.55  # More buying pressure
        volume_bear = taker_ratio[i] < 0.45  # More selling pressure
        
        # === RSI PULLBACK ZONES ===
        # Long: RSI pulled back to 40-55 in uptrend (not oversold, just pullback)
        rsi_pullback_long = (rsi[i] >= 40.0) and (rsi[i] <= 55.0)
        # Short: RSI pulled back to 45-60 in downtrend (not overbought, just retracement)
        rsi_pullback_short = (rsi[i] >= 45.0) and (rsi[i] <= 60.0)
        
        # === COOLDOWN CHECK ===
        bars_since_trade = i - last_trade_bar
        can_trade = bars_since_trade >= cooldown_bars
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 1d bull + RSI pullback + volume + session + cooldown
        if (htf_4h_bull and htf_1d_bull and rsi_pullback_long and 
            volume_bull and in_session and can_trade and above_sma200):
            # Size boost when all HTF aligned
            desired_signal = SIZE_STRONG
        
        # SHORT: 4h bear + 1d bear + RSI pullback + volume + session + cooldown
        elif (htf_4h_bear and htf_1d_bear and rsi_pullback_short and 
              volume_bear and in_session and can_trade and below_sma200):
            desired_signal = -SIZE_STRONG
        
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
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals