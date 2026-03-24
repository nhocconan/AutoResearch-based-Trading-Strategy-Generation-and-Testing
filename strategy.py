#!/usr/bin/env python3
"""
Experiment #338: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback v1

Hypothesis: Previous experiments failed due to OVER-FILTERING (too many confluence requirements = 0 trades).
Return to PROVEN pattern: HMA trend + RSI pullback + HTF confirmation, but SIMPLIFIED for 4h timeframe.

Key learnings from 300+ failed experiments:
1. Complex regime switching (CHOP/CRSI) consistently fails on BTC/ETH
2. Too many filters = 0 trades = auto-reject
3. 4h timeframe should target 20-50 trades/year (not 100+)
4. HMA trend + RSI pullback is the most reliable pattern for crypto

Strategy Logic (SIMPLIFIED):
- 1d HMA(50) = major trend bias (only trade in direction)
- 4h HMA(21) = entry trigger on pullback
- RSI(14) = pullback confirmation (35-45 for long, 55-65 for short)
- ATR(14) = stoploss at 2.5x from entry

Entry Conditions (LOOSENED for trades):
- Long: 1d HMA bull + 4h HMA bull + RSI 35-50 + price pullback to HMA
- Short: 1d HMA bear + 4h HMA bear + RSI 50-65 + price rally to HMA

Position sizing: 0.25 base, 0.30 when 1d strongly aligned
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_trend_rsi_pullback_1d_v1"
timeframe = "4h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA50) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === HMA SLOPE (trend strength) ===
        hma_slope_bull = False
        hma_slope_bear = False
        if i > 5 and not np.isnan(hma_4h[i-5]):
            hma_slope_bull = hma_4h[i] > hma_4h[i-5]
            hma_slope_bear = hma_4h[i] < hma_4h[i-5]
        
        # === RSI PULLBACK ZONES (LOOSENED for more trades) ===
        # Long: RSI pulled back to 35-50 in uptrend
        rsi_pullback_long = 35.0 <= rsi[i] <= 52.0
        # Short: RSI rallied to 48-65 in downtrend
        rsi_pullback_short = 48.0 <= rsi[i] <= 65.0
        
        # === PRICE VS HMA (pullback to support/resistance) ===
        # Long: price near or slightly below HMA (pullback)
        price_near_hma_long = close[i] <= hma_4h[i] * 1.015  # within 1.5% above HMA
        price_near_hma_long = price_near_hma_long and close[i] >= hma_4h[i] * 0.97  # not too far below
        
        # Short: price near or slightly above HMA (rally)
        price_near_hma_short = close[i] >= hma_4h[i] * 0.985  # within 1.5% below HMA
        price_near_hma_short = price_near_hma_short and close[i] <= hma_4h[i] * 1.03  # not too far above
        
        # === SMA200 FILTER (major trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer filters) ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1d bull + 4h bull + RSI pullback + price near HMA
        if htf_bull and hma_bull and rsi_pullback_long and price_near_hma_long:
            # Require SMA200 confirmation OR strong 1d alignment
            if above_sma200 or (close[i] > hma_1d_aligned[i] * 1.02):
                desired_signal = SIZE_STRONG if hma_slope_bull else SIZE_BASE
        
        # SHORT ENTRY: 1d bear + 4h bear + RSI pullback + price near HMA
        elif htf_bear and hma_bear and rsi_pullback_short and price_near_hma_short:
            # Require SMA200 confirmation OR strong 1d alignment
            if below_sma200 or (close[i] < hma_1d_aligned[i] * 0.98):
                desired_signal = -SIZE_STRONG if hma_slope_bear else -SIZE_BASE
        
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
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals