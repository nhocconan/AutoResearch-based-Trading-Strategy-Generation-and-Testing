#!/usr/bin/env python3
"""
Experiment #256: 12h Primary + 1d HTF — Donchian Breakout with HMA Trend

Hypothesis: After 200+ failed experiments with complex regime-switching (CHOP + CRSI),
return to proven momentum breakout strategy that generates consistent trades:
- 1d HMA(21) for macro trend bias (proven in best strategies)
- 12h Donchian(20) breakout for entry trigger (catches momentum moves)
- 12h RSI(14) filter at 40/70 long / 30/60 short (avoids extremes, frequent triggers)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.25 (conservative for 12h volatility)

KEY INSIGHT FROM FAILURES:
- #244, #249, #252, #253, #254: Choppiness Index creates whipsaws and negative Sharpe
- #248, #250, #255: 0 trades from too-strict entry conditions (CRSI 15/85 too rare)
- #251: Simple HMA+RSI worked but needs momentum trigger for better entries
- SOLUTION: Remove CHOP, use Donchian breakout + simple RSI filter + 1d trend

TARGET: 20-50 trades/year on 12h, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
DONCHIAN(20) on 12h = 10-day lookback, breakouts occur 2-4x/month = 48-96/year
With 1d HMA filter, expect 30-60 trades/year (within target range)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_rsi_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate 1d HMA for macro trend (aligned properly with shift(1))
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_21[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT (use previous bar's levels to avoid look-ahead) ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === RSI FILTER (avoid extremes) ===
        rsi_ok_long = rsi_14[i] >= 40.0 and rsi_14[i] <= 70.0
        rsi_ok_short = rsi_14[i] >= 30.0 and rsi_14[i] <= 60.0
        
        # === CHECK EXITS FIRST ===
        exit_triggered = False
        
        # Stoploss check
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                exit_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                exit_triggered = True
        
        # Trend reversal exit
        if in_position and position_side > 0 and price_below_hma_1d:
            exit_triggered = True
        
        if in_position and position_side < 0 and price_above_hma_1d:
            exit_triggered = True
        
        # === DETERMINE SIGNAL ===
        desired_signal = 0.0
        
        if exit_triggered:
            desired_signal = 0.0
        elif in_position:
            # Hold position if trend still valid
            if position_side > 0 and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        else:
            # New entry signals
            if price_above_hma_1d and breakout_long and rsi_ok_long:
                desired_signal = POSITION_SIZE
            elif price_below_hma_1d and breakout_short and rsi_ok_short:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals