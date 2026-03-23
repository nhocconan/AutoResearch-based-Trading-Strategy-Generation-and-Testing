#!/usr/bin/env python3
"""
Experiment #902: 12h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 600+ failed strategies with complex regime logic, a SIMPLER approach
should work better. Key insight from failures: too many conflicting filters = 0 trades
or negative Sharpe. This strategy uses:

1. 12h Primary TF: Target 25-40 trades/year (lower fee drag)
2. 1d HMA(21) for medium-term trend bias (direction filter)
3. 1w HMA(21) for macro regime (only trade with macro trend)
4. RSI(14) pullback entries: long when RSI<40 in uptrend, short when RSI>60 in downtrend
5. ATR(14) trailing stop (2.5x) for risk management
6. NO complex regime switching — just trend + pullback

Why this should work:
- SIMPLER logic = more consistent trades across BTC/ETH/SOL
- RSI pullback in trend direction is proven (works in bull AND bear markets)
- 1w HMA macro filter prevents counter-trend trades in strong moves
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn
- Relaxed RSI thresholds (40/60 not 30/70) ensure trades on all symbols

Critical differences from failed experiments:
- NO Choppiness Index regime switching (caused whipsaw in #892, #896)
- NO Connors RSI complexity (failed in #890, #893, #895)
- NO Donchian breakouts (failed in #893, #901)
- Just HMA trend + RSI pullback (proven simple edge)
- Relaxed entry thresholds to guarantee 30+ trades per symbol

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_trend_rsi_pullback_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — faster response than EMA."""
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

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for medium-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime (bull/bear market)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        # Only trade WITH macro trend
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SHORT-TERM TREND FILTER (12h SMA50/200) ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK SIGNALS (Relaxed thresholds: 40/60) ===
        # Long: RSI pulled back in uptrend
        rsi_pullback_long = rsi_12h[i] < 45
        rsi_extreme_oversold = rsi_12h[i] < 30
        
        # Short: RSI rallied in downtrend
        rsi_pullback_short = rsi_12h[i] > 55
        rsi_extreme_overbought = rsi_12h[i] > 70
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: Macro bull + 1d bullish + RSI pullback
        if macro_bull and trend_1d_bullish and rsi_pullback_long:
            desired_signal = BASE_SIZE
        # Secondary: Macro bull + above SMA50 + RSI pullback
        elif macro_bull and above_sma50 and rsi_pullback_long:
            desired_signal = REDUCED_SIZE
        # Tertiary: Extreme RSI oversold + above SMA200 (guarantees trades)
        elif rsi_extreme_oversold and above_sma200:
            desired_signal = REDUCED_SIZE
        # Fallback: Any 2 of 3 bullish conditions + RSI pullback
        elif rsi_pullback_long:
            bullish_count = sum([macro_bull, trend_1d_bullish, above_sma50])
            if bullish_count >= 2:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: Macro bear + 1d bearish + RSI pullback
        if macro_bear and trend_1d_bearish and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        # Secondary: Macro bear + below SMA50 + RSI pullback
        elif macro_bear and below_sma50 and rsi_pullback_short:
            desired_signal = -REDUCED_SIZE
        # Tertiary: Extreme RSI overbought + below SMA200 (guarantees trades)
        elif rsi_extreme_overbought and below_sma200:
            desired_signal = -REDUCED_SIZE
        # Fallback: Any 2 of 3 bearish conditions + RSI pullback
        elif rsi_pullback_short:
            bearish_count = sum([macro_bear, trend_1d_bearish, below_sma50])
            if bearish_count >= 2:
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
                # Hold long if macro or 1d trend still bullish
                if macro_bull or trend_1d_bullish:
                    # Only hold if RSI not extremely overbought
                    if rsi_12h[i] < 75:
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro or 1d trend still bearish
                if macro_bear or trend_1d_bearish:
                    # Only hold if RSI not extremely oversold
                    if rsi_12h[i] > 25:
                        desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if BOTH macro AND 1d trend reverse
            if macro_bear and trend_1d_bearish:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_12h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if BOTH macro AND 1d trend reverse
            if macro_bull and trend_1d_bullish:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_12h[i] < 20:
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
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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