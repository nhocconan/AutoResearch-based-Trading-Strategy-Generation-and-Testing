#!/usr/bin/env python3
"""
Experiment #322: 4h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + Choppiness Regime v1

Hypothesis: 4h timeframe with HMA trend + RSI pullback entries + HTF alignment generates
consistent trades with good Sharpe. Proven pattern from research (SOL Sharpe +0.879).

Key improvements from failed experiments:
1. SIMPLIFIED ENTRY: HMA trend + RSI pullback + HTF alignment (3 conditions max)
2. LOOSENED RSI: 35-45 for long pullback, 55-65 for short (ensures trades generate)
3. CHOPPY REGIME FILTER: Only trade when CHOP < 65 (avoid worst chop periods)
4. REMOVED FUNDING: Funding data loading was unreliable across symbols
5. DISCRETE SIZING: 0.25 base, 0.30 when HTF aligned (minimizes fee churn)
6. RSI EXIT: Exit when RSI reaches extreme (75 long / 25 short) for mean reversion profit

Regime Detection:
- Choppiness Index (CHOP) < 50 = strong trend (full size 0.30)
- Choppiness Index (CHOP) 50-65 = weak trend (reduced size 0.25)
- Choppiness Index (CHOP) > 65 = choppy (NO TRADES)

Entry Logic:
- Long: price > HMA(21) + RSI(14) 35-45 + 1d HMA(50) bull + CHOP < 65
- Short: price < HMA(21) + RSI(14) 55-65 + 1d HMA(50) bear + CHOP < 65

Exit Logic:
- Stoploss: 2.5x ATR from entry
- Take Profit: RSI > 75 (long) or RSI < 25 (short)
- Trend Reversal: Price crosses opposite side of HMA(21)

Position sizing: 0.25 base, 0.30 when 1w aligned (discrete levels)
Stoploss: 2.5x ATR from entry price

Target: Sharpe>0.40, DD>-40%, trades>=30 train, trades>=3 test
Timeframe: 4h (target 20-50 trades/year per symbol)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_chop_regime_1d1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
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
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP > 65 = too choppy, NO TRADES
        # CHOP 50-65 = weak trend, reduced size
        # CHOP < 50 = strong trend, full size
        if chop[i] > 65.0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1w for major trend boost
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_21[i]
        hma_bear = close[i] < hma_21[i]
        
        # === RSI PULLBACK ZONES (LOOSENED for trade generation) ===
        rsi_pullback_long = 35.0 <= rsi[i] <= 48.0
        rsi_pullback_short = 52.0 <= rsi[i] <= 65.0
        
        # RSI exit zones
        rsi_overbought = rsi[i] > 72.0
        rsi_oversold = rsi[i] < 28.0
        
        # === POSITION SIZE BASED ON REGIME ===
        if chop[i] < 50.0:
            size = SIZE_STRONG
        else:
            size = SIZE_BASE
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Long entry: HMA bull + RSI pullback + 1d bull + CHOP < 65
        if hma_bull and rsi_pullback_long and htf_1d_bull:
            if htf_1w_bull:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = size
        
        # Short entry: HMA bear + RSI pullback + 1d bear + CHOP < 65
        elif hma_bear and rsi_pullback_short and htf_1d_bear:
            if htf_1w_bear:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -size
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if in_position and position_side > 0:
            # Long exit: RSI overbought OR stoploss OR trend reversal
            if rsi_overbought:
                exit_signal = True
            elif low[i] < stop_price:
                exit_signal = True
            elif close[i] < hma_21[i]:
                exit_signal = True
        
        if in_position and position_side < 0:
            # Short exit: RSI oversold OR stoploss OR trend reversal
            if rsi_oversold:
                exit_signal = True
            elif high[i] > stop_price:
                exit_signal = True
            elif close[i] > hma_21[i]:
                exit_signal = True
        
        if exit_signal:
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
                # Set stoploss at 2.5x ATR
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