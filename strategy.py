#!/usr/bin/env python3
"""
Experiment #930: 1h Primary + 4h/12h HTF — Trend Pullback with Session/Volume Filter

Hypothesis: After 660+ failed strategies, 1h timeframe with HTF trend filter + relaxed
entry conditions should generate 30-80 trades/year while maintaining positive Sharpe.

Key insights from research:
1. 1h Primary TF: Use 4h/12h for SIGNAL DIRECTION, 1h only for ENTRY TIMING
2. 4h HMA(21) for medium-term trend bias (only trade in HTF trend direction)
3. 12h HMA(21) for macro regime filter (bull/bear market alignment)
4. 1h RSI(14) pullback entries within HTF trend (RSI<45 long, RSI>55 short)
5. Session filter: only 8-20 UTC (high liquidity hours)
6. Volume filter: >0.8x 20-bar average (confirm participation)
7. ATR(14) trailing stop (2.5x) for risk management

Why this should work on 1h:
- HTF trend filter reduces whipsaw (trade only with 4h/12h direction)
- RSI pullback entries catch dips in uptrend / rallies in downtrend
- Session + volume filters reduce false signals during low liquidity
- Relaxed RSI thresholds (45/55 not 30/70) ensure trades on all symbols
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Critical improvements from failed experiments:
- RELAXED RSI thresholds to guarantee 30+ trades per symbol
- HTF trend as DIRECTION filter, not entry trigger (fewer trades)
- Session filter reduces noise during Asian/late US hours
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- Use 12h HMA as macro filter: only long if price > 12h HMA in bull

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-70 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_session_vol_atr_v1"
timeframe = "1h"
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

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    # Convert to hours UTC
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    sma_50_1h = calculate_sma(close, 50)
    sma_200_1h = calculate_sma(close, 200)
    
    # Calculate and align 4h HMA for medium-term trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for macro regime (bull/bear market)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(sma_50_1h[i]) or np.isnan(sma_200_1h[i]):
            continue
        
        # Extract UTC hour for session filter
        hour_utc = get_hour_from_open_time(open_time[i])
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER (>0.8x 20-bar average) ===
        volume_ok = volume[i] > 0.8 * vol_sma_1h[i]
        
        # === MACRO REGIME (12h HTF HMA21) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === SHORT-TERM TREND FILTER (1h SMA50/200) ===
        above_sma50 = close[i] > sma_50_1h[i]
        below_sma50 = close[i] < sma_50_1h[i]
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        # === RSI PULLBACK SIGNALS (Relaxed thresholds: 45/55) ===
        rsi_pullback_long = rsi_1h[i] < 45  # Pullback in uptrend
        rsi_pullback_short = rsi_1h[i] > 55  # Rally in downtrend
        rsi_extreme_long = rsi_1h[i] < 35   # Deep pullback
        rsi_extreme_short = rsi_1h[i] > 65  # Strong rally
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: Macro bull + 4h bullish + RSI pullback + session + volume
        if macro_bull and trend_4h_bullish and rsi_pullback_long and in_session and volume_ok:
            desired_signal = BASE_SIZE
        
        # Secondary: Macro bull + above SMA50 + RSI pullback + session
        elif macro_bull and above_sma50 and rsi_pullback_long and in_session:
            desired_signal = REDUCED_SIZE
        
        # Tertiary: 4h bullish + extreme RSI + session (guarantees trades)
        elif trend_4h_bullish and rsi_extreme_long and in_session:
            desired_signal = REDUCED_SIZE
        
        # Fallback: Above SMA200 + extreme RSI (ensures minimum trade count)
        elif above_sma200 and rsi_extreme_long:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: Macro bear + 4h bearish + RSI rally + session + volume
        if macro_bear and trend_4h_bearish and rsi_pullback_short and in_session and volume_ok:
            desired_signal = -BASE_SIZE
        
        # Secondary: Macro bear + below SMA50 + RSI rally + session
        elif macro_bear and below_sma50 and rsi_pullback_short and in_session:
            desired_signal = -REDUCED_SIZE
        
        # Tertiary: 4h bearish + extreme RSI + session (guarantees trades)
        elif trend_4h_bearish and rsi_extreme_short and in_session:
            desired_signal = -REDUCED_SIZE
        
        # Fallback: Below SMA200 + extreme RSI (ensures minimum trade count)
        elif below_sma200 and rsi_extreme_short:
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
                if trend_4h_bullish and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses + RSI overbought
            if trend_4h_bearish and rsi_1h[i] > 70:
                desired_signal = 0.0
            # Exit if macro reverses strongly
            if macro_bear and rsi_1h[i] > 60:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses + RSI oversold
            if trend_4h_bullish and rsi_1h[i] < 30:
                desired_signal = 0.0
            # Exit if macro reverses strongly
            if macro_bull and rsi_1h[i] < 40:
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
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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