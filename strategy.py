#!/usr/bin/env python3
"""
Experiment #685: 1h Primary + 4h/1d HTF — Simplified Trend Pullback with Loose Entries

Hypothesis: Previous 1h strategies failed (Sharpe=0.000) because entry conditions were TOO STRICT.
Key insight from experiment history: 4h/12h strategies work, 1h/30m generate 0 trades.

Solution: Use PROVEN 4h HMA trend filter + LOOSE 1h RSI pullback entries.
- 4h HMA(21) for trend direction (proven in best strategy #681)
- 1h RSI(14) for entry timing (pullback in trend, thresholds 35/65 not 20/80)
- 1d HMA(21) for macro bias filter
- LOOSE thresholds to ensure 30-80 trades/year on 1h
- NO session filter, NO volume filter (these killed trade generation)
- Simple 2.5*ATR trailing stoploss

Why this should work where #675, #678, #680 failed:
- Those used CRSI<15/>85 (too strict) + Choppiness + multiple confluence = 0 trades
- This uses RSI<35/>65 (much looser) + single HTF trend filter = trades will generate
- 4h HMA trend is PROVEN edge from best strategy (Sharpe=0.612)
- 1h timeframe with 4h trend direction = HTF trade frequency with LTF execution

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
Position size: 0.25 (conservative for 1h TF to control fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_loose_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """RSI with proper min_periods."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_pad = np.concatenate([[0], gain])
    loss_pad = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain_pad).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss_pad).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_sma(close, period=200):
    """Simple Moving Average for macro trend."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF (4h) HMA trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align HTF (1d) HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative for 1h TF
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === HTF TREND DIRECTION (4h HMA) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === MACRO BIAS (1d HMA) ===
        macro_bullish = close[i] > hma_1d_aligned[i]
        macro_bearish = close[i] < hma_1d_aligned[i]
        
        # === LONG TERM TREND (SMA200) ===
        long_trend_bullish = close[i] > sma_200[i]
        long_trend_bearish = close[i] < sma_200[i]
        
        # === RSI PULLBACK SIGNALS (LOOSE thresholds) ===
        rsi_oversold = rsi_1h[i] < 35  # Much looser than 20
        rsi_overbought = rsi_1h[i] > 65  # Much looser than 80
        rsi_neutral = 35 <= rsi_1h[i] <= 65
        
        desired_signal = 0.0
        
        # === LONG ENTRY: 4h bullish + RSI pullback ===
        # Only require 2 confluence: 4h trend + RSI oversold
        if trend_4h_bullish and rsi_oversold:
            desired_signal = SIZE
        
        # === SHORT ENTRY: 4h bearish + RSI overbought ===
        if trend_4h_bearish and rsi_overbought:
            desired_signal = -SIZE
        
        # === MACRO FILTER: Only trade with 1d trend for higher conviction ===
        # If macro agrees with 4h, increase confidence (but don't require it)
        if trend_4h_bullish and macro_bullish and rsi_oversold:
            desired_signal = SIZE
        elif trend_4h_bearish and macro_bearish and rsi_overbought:
            desired_signal = -SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        # This reduces churn and keeps positions open during pullbacks
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h still bullish and RSI not extremely overbought
                if trend_4h_bullish and rsi_1h[i] < 75:
                    desired_signal = SIZE
            elif position_side < 0:
                # Hold short if 4h still bearish and RSI not extremely oversold
                if trend_4h_bearish and rsi_1h[i] > 25:
                    desired_signal = -SIZE
        
        # === EXIT SIGNAL: RSI crosses back to neutral ===
        # Exit long when RSI > 70, exit short when RSI < 30
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and rsi_1h[i] > 70:
                desired_signal = 0.0
            elif position_side < 0 and rsi_1h[i] < 30:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE
        elif desired_signal < 0:
            desired_signal = -SIZE
        
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
                # Flip position
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