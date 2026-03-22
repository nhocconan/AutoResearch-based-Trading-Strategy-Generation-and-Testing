#!/usr/bin/env python3
"""
Experiment #426: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Mean Reversion

Hypothesis: After 368 failed experiments, clear patterns emerge:
1. Complex regime detection (CHOP + ADX + multiple filters) causes 0 trades or whipsaw
2. 12h timeframe needs SIMPLE logic with multiple entry paths to ensure trade frequency
3. 1d HMA(21) for major trend bias (not strict filter — allows counter-trend in extremes)
4. RSI(14) extremes (<30/>70) with SMA(200) filter — proven mean reversion edge
5. ATR-based stoploss (2.5x) protects capital in crash scenarios
6. Fewer confluence requirements (1-2 conditions) = more trades = better stats

Why this might beat current best (Sharpe=0.435):
- Simpler logic = fewer false rejections = more trades
- 12h TF has lower fee drag than 4h/1h (fewer trades, same capture)
- 1d HTF provides trend context without blocking all counter-trend trades
- RSI extremes have proven mean reversion edge across all market regimes
- Multiple entry paths ensure >=30 trades/symbol on train

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 25-50 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_sma200_1d_simp_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    hma_12h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(sma_200[i]) or np.isnan(hma_12h_21[i]):
            continue
        
        # === 1D MAJOR TREND (bias, not strict filter) ===
        # Price above 1d HMA = bull bias (favor longs)
        # Price below 1d HMA = bear bias (favor shorts)
        bull_bias = close[i] > hma_1d_21_aligned[i]
        bear_bias = close[i] < hma_1d_21_aligned[i]
        
        # === 12H LOCAL TREND ===
        local_bull = close[i] > hma_12h_21[i]
        local_bear = close[i] < hma_12h_21[i]
        
        # === SMA200 TREND FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI EXTREMES (mean reversion signals) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        
        # === ENTRY LOGIC — MULTIPLE PATHS (ensure trade frequency) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # PATH 1: RSI extreme + trend alignment (strongest signal)
        if bull_bias and above_sma200 and rsi_extreme_oversold:
            new_signal = BASE_SIZE
        elif bear_bias and below_sma200 and rsi_extreme_overbought:
            new_signal = -BASE_SIZE
        
        # PATH 2: RSI moderate + local trend (medium strength)
        if new_signal == 0.0:
            if bull_bias and rsi_oversold and local_bull:
                new_signal = REDUCED_SIZE
            elif bear_bias and rsi_overbought and local_bear:
                new_signal = -REDUCED_SIZE
        
        # PATH 3: Counter-trend on extreme RSI (catch bottoms/tops)
        if new_signal == 0.0:
            if rsi_14[i] < 20.0:  # Very oversold — long regardless of trend
                new_signal = REDUCED_SIZE
            elif rsi_14[i] > 80.0:  # Very overbought — short regardless of trend
                new_signal = -REDUCED_SIZE
        
        # PATH 4: Frequency boost — if no trade for 12 bars, loosen conditions
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            if bull_bias and rsi_oversold:
                new_signal = REDUCED_SIZE * 0.8
            elif bear_bias and rsi_overbought:
                new_signal = -REDUCED_SIZE * 0.8
        
        # === EXIT CONDITIONS ===
        # RSI mean reversion exit (take profit)
        if in_position and position_side > 0 and rsi_14[i] > 60.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 40.0:
            new_signal = 0.0
        
        # Local trend reversal exit
        if in_position and position_side > 0 and local_bear:
            new_signal = 0.0
        if in_position and position_side < 0 and local_bull:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals