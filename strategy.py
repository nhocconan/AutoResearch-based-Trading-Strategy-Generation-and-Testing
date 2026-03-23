#!/usr/bin/env python3
"""
Experiment #894: 4h Primary + 12h/1d HTF — Fisher Transform + HMA Trend + ATR Stops

Hypothesis: After 600+ failed strategies, the key insight is SIMPLICITY + proper MTF.
Recent failures show complex regime logic (CRSI+Chop+Donchian+multiple HTFs) leads to
0 trades or negative Sharpe. This strategy uses:

1. 4h Primary TF: Proven to work better than 12h/1d in recent experiments (#889 Sharpe=0.206)
2. 12h HMA(21) for trend bias (single HTF, not 1d+1w which overcomplicates)
3. Ehlers Fisher Transform(9) for entry timing — catches reversals better than RSI/CRSI
4. Simple trend filter: only long if price > 12h HMA, only short if price < 12h HMA
5. ATR(14) trailing stop (2.5x) for risk management
6. Relaxed Fisher thresholds (-1.5/+1.5) to ensure 30+ trades per symbol

Why Fisher Transform:
- Normalizes price to Gaussian distribution (-1.5 to +1.5 range)
- Catches reversals at extremes better than RSI
- Less tried than CRSI in our experiment history
- Works well in both trending and ranging markets

Why simpler than previous attempts:
- Single HTF (12h) not dual (1d+1w) — reduces conflicting signals
- No Choppiness Index — regime detection was causing 0 trades
- No Donchian breakouts — too many false signals in choppy markets
- Fisher + HMA trend is cleaner confluence

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_12h_atr_trail_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — smoother than EMA, less lag than SMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) * 0.999 + 0.001
    3. Fisher: 0.5 * ln((1 + normalized) / (1 - normalized))
    
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    Exit: Fisher crosses 0 or opposite extreme
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period - 1, n):
        # Typical price
        typical = (high[i] + low[i]) / 2.0
        
        # Highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize to 0-1 range with bounds
        normalized = (typical - lowest) / (highest - lowest) * 0.999 + 0.001
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Previous Fisher value for crossover detection
        if i > 0:
            fisher_prev[i] = fisher[i-1]
        else:
            fisher_prev[i] = 0.0
    
    return fisher, fisher_prev

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

def calculate_rsi(close, period=14):
    """Relative Strength Index — fallback filter."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    fisher_4h, fisher_prev_4h = calculate_fisher_transform(high, low, period=9)
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    
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
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_prev_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(rsi_4h[i]):
            continue
        
        # === TREND BIAS (12h HTF HMA21) ===
        trend_bullish = close[i] > hma_12h_aligned[i]
        trend_bearish = close[i] < hma_12h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_prev_4h[i] < -1.5) and (fisher_4h[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_prev_4h[i] > 1.5) and (fisher_4h[i] <= 1.5)
        
        # === RSI FILTER (avoid extreme overbought/oversold entries against trend) ===
        rsi_not_extreme_long = rsi_4h[i] < 75  # Don't long if already overbought
        rsi_not_extreme_short = rsi_4h[i] > 25  # Don't short if already oversold
        
        # === ENTRY SIGNALS ===
        desired_signal = 0.0
        
        # Long entry: Fisher cross + bullish trend + RSI filter
        if fisher_long_cross and trend_bullish and rsi_not_extreme_long:
            desired_signal = BASE_SIZE
        
        # Short entry: Fisher cross + bearish trend + RSI filter
        if fisher_short_cross and trend_bearish and rsi_not_extreme_short:
            desired_signal = -BASE_SIZE
        
        # Fallback: Fisher extreme alone (ensures trades even if trend filter too strict)
        if desired_signal == 0.0:
            if fisher_4h[i] < -1.8 and trend_bullish:  # Very oversold + trend
                desired_signal = REDUCED_SIZE
            if fisher_4h[i] > 1.8 and trend_bearish:  # Very overbought + trend
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
                # Hold long if trend still bullish and Fisher not overbought
                if trend_bullish and fisher_4h[i] < 1.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend still bearish and Fisher not oversold
                if trend_bearish and fisher_4h[i] > -1.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses OR Fisher very overbought
            if trend_bearish and fisher_4h[i] > 0.5:
                desired_signal = 0.0
            if fisher_4h[i] > 1.5:  # Fisher overbought
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses OR Fisher very oversold
            if trend_bullish and fisher_4h[i] < -0.5:
                desired_signal = 0.0
            if fisher_4h[i] < -1.5:  # Fisher oversold
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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