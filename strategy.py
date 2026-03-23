#!/usr/bin/env python3
"""
Experiment #789: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback + Donchian

Hypothesis: After analyzing 500+ failed strategies and current best (Sharpe=0.612):
1. 4h is the sweet spot timeframe (exp #779 Sharpe=0.223, #786 Sharpe=0.284)
2. Complex regime filters often prevent trades (many strategies = 0 trades)
3. HMA trend + RSI pullback is proven (current best uses similar logic)
4. Donchian breakout provides momentum confirmation for trending moves
5. 1d HMA(21) provides strong trend bias without being too slow
6. Relaxed RSI thresholds (30/70) ensure >=10 trades/train, >=3 trades/test
7. Simple ATR(14) trailing stop at 2.5x protects from major drawdowns
8. Position sizing: 0.25-0.30 discrete levels to control fees

Strategy design:
1. 1d HMA(21) for long-term trend bias (aligned via mtf_data helper)
2. 4h HMA(16/48 crossover) for medium-term trend direction
3. 4h RSI(14) for pullback entries (30/70 thresholds - relaxed)
4. 4h Donchian(20) for breakout confirmation
5. 4h ATR(14) for trailing stop (2.5x)
6. Discrete signals: 0.0, ±0.25, ±0.30
7. LOOSE entry conditions to ensure trades are generated

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_donchian(high, low, period=20):
    """Donchian Channels."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return upper, lower
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    hma_16_4h = calculate_hma(close, 16)
    hma_48_4h = calculate_hma(close, 48)
    rsi_14_4h = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    atr_14_4h = calculate_atr(high, low, close, 14)
    
    # Calculate and align HTF HMA for trend bias
    hma_21_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d_raw)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_16_4h[i]) or np.isnan(hma_48_4h[i]):
            continue
        if np.isnan(rsi_14_4h[i]) or np.isnan(atr_14_4h[i]) or atr_14_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_21_1d_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(donchian_lower[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_21_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_21_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (4h HMA crossover) ===
        hma_bullish = hma_16_4h[i] > hma_48_4h[i]
        hma_bearish = hma_16_4h[i] < hma_48_4h[i]
        
        # === RSI SIGNALS (relaxed thresholds for more trades) ===
        rsi_oversold = rsi_14_4h[i] < 35
        rsi_overbought = rsi_14_4h[i] > 65
        rsi_neutral_low = 35 <= rsi_14_4h[i] <= 50
        rsi_neutral_high = 50 <= rsi_14_4h[i] <= 65
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (LOOSE for trades) ===
        # Primary: 1d bullish + 4h HMA bullish + RSI pullback
        if trend_1d_bullish and hma_bullish:
            if rsi_neutral_low:
                desired_signal = BASE_SIZE
            elif rsi_oversold:
                desired_signal = BASE_SIZE
            elif breakout_long:
                desired_signal = REDUCED_SIZE
        
        # Secondary: 1d bullish + 4h HMA neutral + RSI oversold (mean reversion)
        elif trend_1d_bullish and not hma_bearish:
            if rsi_oversold:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY CONDITIONS (LOOSE for trades) ===
        # Primary: 1d bearish + 4h HMA bearish + RSI pullback
        elif trend_1d_bearish and hma_bearish:
            if rsi_neutral_high:
                desired_signal = -BASE_SIZE
            elif rsi_overbought:
                desired_signal = -BASE_SIZE
            elif breakout_short:
                desired_signal = -REDUCED_SIZE
        
        # Secondary: 1d bearish + 4h HMA neutral + RSI overbought (mean reversion)
        elif trend_1d_bearish and not hma_bullish:
            if rsi_overbought:
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
                # Hold long if 1d trend intact and RSI not extreme overbought
                if trend_1d_bullish and rsi_14_4h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend intact and RSI not extreme oversold
                if trend_1d_bearish and rsi_14_4h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses
            if trend_1d_bearish:
                desired_signal = 0.0
            # Exit if RSI extreme overbought
            if rsi_14_4h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses
            if trend_1d_bullish:
                desired_signal = 0.0
            # Exit if RSI extreme oversold
            if rsi_14_4h[i] < 20:
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
                entry_atr = atr_14_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14_4h[i]
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