#!/usr/bin/env python3
"""
EXPERIMENT #019 - MTF KAMA Trend + Daily Regime + 1h RSI Pullback
==================================================================================================
Hypothesis: Use 1h primary timeframe with 4h KAMA (adaptive trend) + Daily SMA(50) regime filter.
This differs from current best by:
- KAMA instead of Supertrend (adapts to volatility, fewer whipsaws)
- Daily SMA(50) as additional regime filter (avoid counter-trend in strong markets)
- 1h primary instead of 15m (cleaner signals, fewer trades, lower fees)
- Simpler entry logic: RSI pullback in direction of both 4h and daily trend

Why this should work:
- KAMA adapts to market volatility (better than fixed MA in crypto)
- Daily SMA(50) filters out trades against major trend
- 1h timeframe proven to work well with MTF (see #009, #010)
- Fewer signal changes = lower fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_daily_sma_rsi_1h_4h_1d_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        noise = 0.0
        for j in range(1, period):
            noise += abs(close[i - j + 1] - close[i - j])
        
        if noise > 0:
            er[i] = abs(close[i] - close[i - period]) / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant
    sc = np.zeros(n)
    for i in range(n):
        sc[i] = (er[i] * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100.0
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.clip(rsi, 0, 100)
    
    return rsi


def calculate_sma(close, period=50):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False, min_periods=half_period).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean().values
    
    return hma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, period=10, fast=2, slow=30)
    hma_1h = calculate_hma(close, period=21)
    
    # Get 4h data using mtf_data helper (MUST use this for proper alignment)
    df_4h = get_htf_data(prices, '4h')
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # 4h indicators
    kama_4h = calculate_kama(c_4h, period=10, fast=2, slow=30)
    rsi_4h = calculate_rsi(c_4h, period=14)
    atr_4h = calculate_atr(h_4h, l_4h, c_4h, period=14)
    
    # Align 4h indicators to 1h timeframe (auto shift for completed bars)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Get Daily data using mtf_data helper for regime filter
    df_1d = get_htf_data(prices, '1d')
    c_1d = df_1d['close'].values
    
    # Daily indicators
    sma_1d = calculate_sma(c_1d, period=50)
    
    # Align daily indicators to 1h timeframe
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.0875
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45  # Enter when RSI pulls back to 45 in uptrend
    RSI_SHORT_ENTRY = 55  # Enter when RSI pulls back to 55 in downtrend
    RSI_EXIT_LONG = 65  # Exit when RSI reaches 65 (overbought)
    RSI_EXIT_SHORT = 35  # Exit when RSI reaches 35 (oversold)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 50, 30)
    
    # Track position state for stoploss/takeprofit
    in_position = np.zeros(n, dtype=bool)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    entry_atr = np.zeros(n)
    
    for i in range(first_valid, n):
        # Skip if invalid data
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        kama_4h_val = kama_4h_aligned[i] if i < len(kama_4h_aligned) else 0
        rsi_4h_val = rsi_4h_aligned[i] if i < len(rsi_4h_aligned) else 50
        atr_4h_val = atr_4h_aligned[i] if i < len(atr_4h_aligned) else atr_1h[i]
        sma_1d_val = sma_1d_aligned[i] if i < len(sma_1d_aligned) else 0
        
        # Current price
        price = close[i]
        
        # Determine 4h trend direction (price vs KAMA)
        trend_4h = 0
        if kama_4h_val > 0 and price > kama_4h_val * 1.001:
            trend_4h = 1
        elif kama_4h_val > 0 and price < kama_4h_val * 0.999:
            trend_4h = -1
        
        # Determine daily regime (price vs SMA50)
        regime_1d = 0
        if sma_1d_val > 0 and price > sma_1d_val * 1.002:
            regime_1d = 1
        elif sma_1d_val > 0 and price < sma_1d_val * 0.998:
            regime_1d = -1
        
        # Check stoploss for existing positions
        if in_position[i - 1]:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1]
            prev_atr = entry_atr[i - 1]
            
            # Calculate stoploss price
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * prev_atr
                
                # Check stoploss
                if price < stoploss_price:
                    signals[i] = 0.0
                    in_position[i] = False
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Check RSI exit (overbought)
                if rsi_1h[i] >= RSI_EXIT_LONG:
                    signals[i] = 0.0
                    in_position[i] = False
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Check trend reversal
                if trend_4h == -1 or regime_1d == -1:
                    signals[i] = 0.0
                    in_position[i] = False
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Hold position
                signals[i] = SIZE_FULL
                in_position[i] = True
                position_side[i] = 1
                entry_price[i] = prev_entry
                entry_atr[i] = prev_atr
                continue
                
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * prev_atr
                
                # Check stoploss
                if price > stoploss_price:
                    signals[i] = 0.0
                    in_position[i] = False
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Check RSI exit (oversold)
                if rsi_1h[i] <= RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    in_position[i] = False
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Check trend reversal
                if trend_4h == 1 or regime_1d == 1:
                    signals[i] = 0.0
                    in_position[i] = False
                    position_side[i] = 0
                    entry_price[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Hold position
                signals[i] = -SIZE_FULL
                in_position[i] = True
                position_side[i] = -1
                entry_price[i] = prev_entry
                entry_atr[i] = prev_atr
                continue
        
        # Entry logic: All timeframes must agree
        # Long entry: 4h uptrend + Daily above SMA + RSI pullback
        if trend_4h == 1 and regime_1d == 1:
            if rsi_1h[i] <= RSI_LONG_ENTRY and rsi_1h[i] >= 30:
                signals[i] = SIZE_FULL
                in_position[i] = True
                position_side[i] = 1
                entry_price[i] = price
                entry_atr[i] = atr_1h[i]
            else:
                signals[i] = 0.0
                in_position[i] = False
        
        # Short entry: 4h downtrend + Daily below SMA + RSI pullback
        elif trend_4h == -1 and regime_1d == -1:
            if rsi_1h[i] >= RSI_SHORT_ENTRY and rsi_1h[i] <= 70:
                signals[i] = -SIZE_FULL
                in_position[i] = True
                position_side[i] = -1
                entry_price[i] = price
                entry_atr[i] = atr_1h[i]
            else:
                signals[i] = 0.0
                in_position[i] = False
        
        # No position
        else:
            signals[i] = 0.0
            in_position[i] = False
    
    return signals