#!/usr/bin/env python3
"""
Experiment #675: 6h Primary + 1d/1w HTF — Regime-Adaptive RSI + HMA Trend

Hypothesis: 6h timeframe sits between 4h and 12h - enough bars for mean reversion,
few enough for trend persistence. Key insight from failed 6h experiments: TOO MANY
filters = 0 trades. This strategy uses LOOSE entry conditions with HTF bias only
as a tiebreaker, not a hard filter.

Key innovations:
1. 1d HMA(21) for major trend bias - but NOT a hard filter (allows counter-trend trades)
2. 6h RSI(14) extremes - primary entry trigger (RSI<30 long, RSI>70 short)
3. 6h SMA(50) slope - secondary confirmation (price position relative to SMA)
4. 1w HMA(21) for meta-regime - determines if we're in bull/bear market
5. Regime-adaptive sizing - larger positions when 1w and 1d agree
6. ATR(14) trailing stop - 2.5x for risk management

Entry conditions (LOOSE to ensure trades):
- LONG: RSI<35 OR (RSI<50 + price>SMA50 + 1d HMA bullish)
- SHORT: RSI>65 OR (RSI>50 + price<SMA50 + 1d HMA bearish)
- 1w HMA determines base bias but doesn't block entries

Target: Sharpe>0.40, trades>=30 train, trades>=3 test
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_rsi_hma_1d1w_v2"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0.0)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    rs[:] = np.nan
    mask = avg_loss > 1e-10
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[avg_loss <= 1e-10] = 100.0
    
    return rsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_roc(close, period=10):
    """Rate of Change"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.zeros(n)
    roc[:] = np.nan
    for i in range(period, n):
        if close[i - period] > 1e-10:
            roc[i] = 100.0 * (close[i] - close[i - period]) / close[i - period]
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    rsi = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, period=50)
    atr = calculate_atr(high, low, close, period=14)
    roc_10 = calculate_roc(close, period=10)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        htf_1d_bear = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Meta-regime: both 1d and 1w agree = strong regime
        regime_bull = htf_1d_bull and htf_1w_bull
        regime_bear = htf_1d_bear and htf_1w_bear
        
        # === 6h RSI SIGNALS ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_extreme_oversold = rsi[i] < 25.0
        rsi_extreme_overbought = rsi[i] > 75.0
        
        # === PRICE vs SMA50 ===
        price_above_sma = close[i] > sma_50[i]
        price_below_sma = close[i] < sma_50[i]
        
        # === MOMENTUM (ROC) ===
        momentum_positive = not np.isnan(roc_10[i]) and roc_10[i] > 0.0
        momentum_negative = not np.isnan(roc_10[i]) and roc_10[i] < 0.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS - prioritize trade frequency) ===
        desired_signal = 0.0
        signal_strength = 0
        
        # LONG entries (multiple paths to entry)
        long_score = 0
        
        # Path 1: Extreme RSI oversold (mean reversion)
        if rsi_extreme_oversold:
            long_score += 3
        
        # Path 2: RSI oversold + price above SMA (pullback in uptrend)
        if rsi_oversold and price_above_sma:
            long_score += 2
        
        # Path 3: RSI neutral + strong HTF bias + momentum
        if 35 <= rsi[i] <= 50 and regime_bull and momentum_positive:
            long_score += 2
        
        # Path 4: Simple RSI < 35 (loose entry)
        if rsi_oversold:
            long_score += 1
        
        # SHORT entries (multiple paths to entry)
        short_score = 0
        
        # Path 1: Extreme RSI overbought (mean reversion)
        if rsi_extreme_overbought:
            short_score += 3
        
        # Path 2: RSI overbought + price below SMA (rally in downtrend)
        if rsi_overbought and price_below_sma:
            short_score += 2
        
        # Path 3: RSI neutral + strong HTF bias + negative momentum
        if 50 <= rsi[i] <= 65 and regime_bear and momentum_negative:
            short_score += 2
        
        # Path 4: Simple RSI > 65 (loose entry)
        if rsi_overbought:
            short_score += 1
        
        # Determine signal based on scores
        if long_score >= 2 and long_score > short_score:
            if regime_bull:
                desired_signal = SIZE_STRONG
                signal_strength = 2
            else:
                desired_signal = SIZE_BASE
                signal_strength = 1
        elif short_score >= 2 and short_score > long_score:
            if regime_bear:
                desired_signal = -SIZE_STRONG
                signal_strength = 2
            else:
                desired_signal = -SIZE_BASE
                signal_strength = 1
        elif long_score >= 3:
            # Extreme oversold always triggers
            desired_signal = SIZE_BASE
            signal_strength = 1
        elif short_score >= 3:
            # Extreme overbought always triggers
            desired_signal = -SIZE_BASE
            signal_strength = 1
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals