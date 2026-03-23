#!/usr/bin/env python3
"""
Experiment #956: 12h Primary + 1d HTF — Donchian Breakout + HMA Trend + RSI Filter

Hypothesis: After 664 failed strategies, simpler is better. 12h timeframe with 1d HTF
trend bias should generate 20-50 trades/year with lower fee drag. Key insight from
research: Donchian breakout + HMA trend + RSI filter worked on SOL (Sharpe +0.782).

Why this should work:
1. 12h timeframe = fewer trades, less fee drag (target 25-40 trades/year)
2. 1d HMA(21) for macro trend bias (proven in multiple experiments)
3. Donchian(20) breakout catches momentum moves in both directions
4. RSI(14) filter avoids entering at extremes (prevents buying tops/selling bottoms)
5. Choppiness Index regime switch: trend-follow in low chop, mean-revert in high chop
6. LOOSENED entry conditions to GUARANTEE trades (learned from 0-trade failures)

Critical improvements over failed experiments:
- Fewer confluence requirements (max 2-3 filters, not 5+)
- Funding rate as OPTIONAL boost, not required
- Relaxed RSI thresholds (30/70 not 25/75)
- Relaxed Choppiness thresholds (50/60 not 45/55)
- Hold logic maintains position through minor pullbacks
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Timeframe: 12h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_rsi_chop_1d_regime_v1"
timeframe = "12h"
leverage = 1.0

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index — measures market choppy vs trending."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        mid[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower, donch_mid = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]) or np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        # === MACRO TREND BIAS (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (12h Choppiness Index) ===
        # Relaxed thresholds to ensure signals generate
        ranging_regime = chop_12h[i] > 50
        trending_regime = chop_12h[i] < 60
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donch_breakout_long = close[i] > donch_upper[i]
        donch_breakout_short = close[i] < donch_lower[i]
        
        # === RSI FILTERS (relaxed thresholds) ===
        rsi_neutral = 35 < rsi_12h[i] < 65
        rsi_oversold = rsi_12h[i] < 40
        rsi_overbought = rsi_12h[i] > 60
        rsi_not_extreme_long = rsi_12h[i] < 75  # Don't buy at extreme overbought
        rsi_not_extreme_short = rsi_12h[i] > 25  # Don't sell at extreme oversold
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 60) — Trend Following with Donchian ===
        if trending_regime:
            # Long: Donchian breakout + RSI not extreme + macro bull bias
            if donch_breakout_long and rsi_not_extreme_long:
                if macro_bull:
                    desired_signal = BASE_SIZE
                else:
                    desired_signal = REDUCED_SIZE  # Counter-trend but breakout valid
            
            # Short: Donchian breakout + RSI not extreme + macro bear bias
            if donch_breakout_short and rsi_not_extreme_short:
                if macro_bear:
                    desired_signal = -BASE_SIZE
                else:
                    desired_signal = -REDUCED_SIZE  # Counter-trend but breakout valid
        
        # === RANGING REGIME (CHOP > 50) — Mean Reversion ===
        if ranging_regime:
            # Long: Price near Donchian lower + RSI oversold
            if close[i] < donch_lower[i] * 1.02 and rsi_oversold:
                desired_signal = REDUCED_SIZE
            
            # Short: Price near Donchian upper + RSI overbought
            if close[i] > donch_upper[i] * 0.98 and rsi_overbought:
                desired_signal = -REDUCED_SIZE
            
            # Additional: RSI extreme mean reversion (guarantees trades)
            if rsi_12h[i] < 30 and macro_bull:
                desired_signal = max(desired_signal, REDUCED_SIZE)
            if rsi_12h[i] > 70 and macro_bear:
                desired_signal = min(desired_signal, -REDUCED_SIZE)
        
        # === NEUTRAL REGIME (50 <= CHOP <= 60) — Conservative ===
        if not trending_regime and not ranging_regime:
            # Only enter with strong confluence
            if donch_breakout_long and macro_bull and rsi_neutral:
                desired_signal = BASE_SIZE
            if donch_breakout_short and macro_bear and rsi_neutral:
                desired_signal = -BASE_SIZE
            
            # RSI extremes as backup
            if rsi_12h[i] < 28:
                desired_signal = max(desired_signal, REDUCED_SIZE)
            if rsi_12h[i] > 72:
                desired_signal = min(desired_signal, -REDUCED_SIZE)
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro trend intact and RSI not overbought
                if macro_bull and rsi_12h[i] < 70:
                    desired_signal = BASE_SIZE
                elif rsi_12h[i] < 65:  # Weaker hold condition
                    desired_signal = REDUCED_SIZE
            elif position_side < 0:
                # Hold short if macro trend intact and RSI not oversold
                if macro_bear and rsi_12h[i] > 30:
                    desired_signal = -BASE_SIZE
                elif rsi_12h[i] > 35:  # Weaker hold condition
                    desired_signal = -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses + RSI overbought
            if macro_bear and rsi_12h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses + RSI oversold
            if macro_bull and rsi_12h[i] < 35:
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