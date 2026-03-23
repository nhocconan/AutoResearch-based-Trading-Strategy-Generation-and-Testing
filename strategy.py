#!/usr/bin/env python3
"""
Experiment #918: 30m Primary + 4h/1d HTF — Simplified Trend Pullback with RSI

Hypothesis: After 600+ failed strategies, the key insight is SIMPLICITY + GUARANTEED TRADES.
Complex regime detection (Choppiness, Fisher, etc.) keeps failing with 0 trades or negative Sharpe.

New approach for 30m:
1. 4h HMA(21) for PRIMARY trend direction (HTF bias)
2. 1d HMA(21) for MACRO regime filter (bull/bear market)
3. 30m RSI(14) pullback entries WITHIN the HTF trend (not extremes)
4. Volume filter (loose: >0.5x avg) — not too strict
5. Session filter (8-20 UTC) for liquidity
6. ATR(14) trailing stop 2.5x for risk management

CRITICAL DIFFERENCE from failed experiments:
- RELAXED RSI thresholds (35-45 for long, 55-65 for short) — NOT extreme 20/80
- This ensures trades happen even in moderate trends
- 4h trend + 30m pullback = proven pattern from best strategy (mtf_4h_triple_regime)
- Fewer filters = more trades (target 40-80/year on 30m)

Why 30m can work:
- Use 4h/1d for DIRECTION (HTF signals are stronger)
- Use 30m only for ENTRY TIMING (when to pull trigger within HTF trend)
- This gives HTF trade frequency with 30m execution precision

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 40-80 trades/year with strict entry filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h1d_hma_vol_session_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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
    """Volume Simple Moving Average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    hours = (prices['open_time'].values // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (30m) indicators
    rsi_30m = calculate_rsi(close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    sma_50_30m = calculate_sma(close, 50)
    sma_200_30m = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro regime (bull/bear market)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Get session hours
    hours = get_hour_from_open_time(prices)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50_30m[i]) or np.isnan(sma_200_30m[i]):
            continue
        if np.isnan(vol_sma_30m[i]) or vol_sma_30m[i] <= 1e-10:
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SHORT-TERM TREND FILTER (30m SMA50/200) ===
        above_sma50 = close[i] > sma_50_30m[i]
        below_sma50 = close[i] < sma_50_30m[i]
        above_sma200 = close[i] > sma_200_30m[i]
        below_sma200 = close[i] < sma_200_30m[i]
        
        # === VOLUME FILTER (loose: >0.5x average) ===
        volume_ok = volume[i] > 0.5 * vol_sma_30m[i]
        
        # === SESSION FILTER (8-20 UTC for liquidity) ===
        session_ok = 8 <= hours[i] <= 20
        
        # === RSI PULLBACK SIGNALS (RELAXED thresholds for trades) ===
        # Long: RSI pullback to 35-45 in uptrend (not extreme oversold)
        rsi_pullback_long = 35 <= rsi_30m[i] <= 50
        # Short: RSI pullback to 50-65 in downtrend (not extreme overbought)
        rsi_pullback_short = 50 <= rsi_30m[i] <= 65
        
        # Extreme RSI for counter-trend entries (guarantees trades)
        rsi_extreme_oversold = rsi_30m[i] < 30
        rsi_extreme_overbought = rsi_30m[i] > 70
        
        desired_signal = 0.0
        
        # === PRIMARY LOGIC: Trend Following with RSI Pullback ===
        # Long: 4h bullish + 1d bullish + RSI pullback + volume + session
        if trend_4h_bullish and macro_bull and rsi_pullback_long and volume_ok:
            if session_ok:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE  # Off-session = smaller size
        
        # Short: 4h bearish + 1d bearish + RSI pullback + volume + session
        if trend_4h_bearish and macro_bear and rsi_pullback_short and volume_ok:
            if session_ok:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -REDUCED_SIZE  # Off-session = smaller size
        
        # === SECONDARY LOGIC: Single HTF trend + extreme RSI (guarantees trades) ===
        if desired_signal == 0.0:
            # Long: 4h bullish + extreme RSI oversold
            if trend_4h_bullish and rsi_extreme_oversold and volume_ok:
                desired_signal = REDUCED_SIZE
            
            # Short: 4h bearish + extreme RSI overbought
            if trend_4h_bearish and rsi_extreme_overbought and volume_ok:
                desired_signal = -REDUCED_SIZE
        
        # === TERTIARY LOGIC: SMA50/200 alignment + RSI (more trade opportunities) ===
        if desired_signal == 0.0:
            # Long: Above SMA50 + Above SMA200 + RSI pullback
            if above_sma50 and above_sma200 and rsi_pullback_long and volume_ok:
                desired_signal = REDUCED_SIZE
            
            # Short: Below SMA50 + Below SMA200 + RSI pullback
            if below_sma50 and below_sma200 and rsi_pullback_short and volume_ok:
                desired_signal = -REDUCED_SIZE
        
        # === QUATERNARY LOGIC: Macro-only + extreme RSI (fallback for trades) ===
        if desired_signal == 0.0:
            # Long: Macro bull + extreme RSI
            if macro_bull and rsi_extreme_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: Macro bear + extreme RSI
            if macro_bear and rsi_extreme_overbought:
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
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_30m[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses
            if trend_4h_bearish and rsi_30m[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses
            if trend_4h_bullish and rsi_30m[i] < 35:
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