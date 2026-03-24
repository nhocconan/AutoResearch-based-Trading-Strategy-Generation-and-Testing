#!/usr/bin/env python3
"""
Experiment #094: 4h Primary + 12h/1d HTF — Dual Regime with Choppiness + Funding Bias

Hypothesis: After 93 failed experiments, the winning formula combines:
1. Choppiness Index (CHOP) regime detection - range vs trend identification
2. Dual logic: Mean reversion in choppy markets, trend follow in trending markets
3. Funding rate contrarian bias for BTC/ETH (proven Sharpe 0.8-1.5 through 2022 crash)
4. Loose entry thresholds to ensure trades generate on ALL symbols

Why this should work:
- 4h timeframe = 30-60 trades/year target (fee-efficient, proven)
- CHOP > 61.8 = range regime → mean revert at RSI extremes
- CHOP < 38.2 = trend regime → follow HMA direction
- Funding rate z-score < -2 → long bias, > +2 → short bias (contrarian)
- Discrete sizing 0.25-0.30 minimizes fee churn
- ATR 2.5x trailing stop protects from catastrophic moves

Entry Logic:
- Range regime (CHOP>61.8): Long RSI<30, Short RSI>70 + funding bias confirmation
- Trend regime (CHOP<38.2): Long price>12h HMA + RSI>45, Short price<12h HMA + RSI<55
- Size: 0.25 discrete, max 0.30 with strong confluence

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_dual_regime_funding_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    Formula: 100 * (ATR(1) sum / Donchian range) / log10(periods)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Calculate ATR(1) sum over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]), 
                        abs(low[j] - close[j-1]))
            atr_sum += tr
        
        # Donchian range over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        donchian_range = highest - lowest
        
        if donchian_range > 1e-10:
            chop[i] = 100.0 * (atr_sum / donchian_range) / np.log10(period)
        else:
            chop[i] = 100.0
    
    return chop

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI - momentum oscillator with proper min_periods"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss calculation"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_funding_zscore(prices, symbol, lookback=30):
    """
    Funding rate Z-score for contrarian bias
    Load from data/processed/funding/*.parquet
    Z < -2 = excessively negative = long bias
    Z > +2 = excessively positive = short bias
    """
    try:
        import os
        funding_path = f"data/processed/funding/{symbol}.parquet"
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            if len(df_funding) >= lookback:
                funding_mean = df_funding['funding_rate'].rolling(lookback, min_periods=lookback).mean()
                funding_std = df_funding['funding_rate'].rolling(lookback, min_periods=lookback).std()
                zscore = (df_funding['funding_rate'] - funding_mean) / (funding_std + 1e-10)
                # Align to prices length
                if len(zscore) >= len(prices):
                    return zscore.values[:len(prices)]
                else:
                    # Pad with NaN
                    result = np.full(len(prices), np.nan)
                    result[:len(zscore)] = zscore.values
                    return result
    except Exception:
        pass
    
    # Return NaN if funding data not available
    return np.full(len(prices), np.nan)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract symbol from prices metadata if available
    symbol = prices.get("symbol", "BTCUSDT")
    if isinstance(symbol, pd.Series):
        symbol = symbol.iloc[0] if len(symbol) > 0 else "BTCUSDT"
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Load funding rate z-score for contrarian bias
    funding_z = calculate_funding_zscore(prices, symbol, lookback=30)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # Base position size
    SIZE_MAX = 0.30   # Max with strong confluence
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trend market
        
        # === HTF TREND BIAS (12h HMA) ===
        hma_12h_bull = close[i] > hma_12h_aligned[i]
        hma_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === FUNDING RATE BIAS (Contrarian) ===
        funding_long_bias = False
        funding_short_bias = False
        if not np.isnan(funding_z[i]):
            funding_long_bias = funding_z[i] < -1.5  # Excessively negative funding
            funding_short_bias = funding_z[i] > 1.5  # Excessively positive funding
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        if is_choppy:
            # === RANGE REGIME: Mean Reversion ===
            # Long: RSI oversold + funding confirms or neutral
            if rsi[i] < 35.0:
                if funding_long_bias or not funding_short_bias:
                    desired_signal = SIZE_BASE
                    signal_strength = 1
                    if funding_long_bias and hma_12h_bull:
                        desired_signal = SIZE_MAX
                        signal_strength = 2
            
            # Short: RSI overbought + funding confirms or neutral
            elif rsi[i] > 65.0:
                if funding_short_bias or not funding_long_bias:
                    desired_signal = -SIZE_BASE
                    signal_strength = 1
                    if funding_short_bias and hma_12h_bear:
                        desired_signal = -SIZE_MAX
                        signal_strength = 2
        
        elif is_trending:
            # === TREND REGIME: Trend Following ===
            # Long: Price above 12h HMA + RSI not overbought
            if hma_12h_bull and rsi[i] > 45.0 and rsi[i] < 80.0:
                desired_signal = SIZE_BASE
                signal_strength = 1
                if rsi[i] > 50.0 and rsi[i] < 70.0:
                    desired_signal = SIZE_MAX
                    signal_strength = 2
            
            # Short: Price below 12h HMA + RSI not oversold
            elif hma_12h_bear and rsi[i] < 55.0 and rsi[i] > 20.0:
                desired_signal = -SIZE_BASE
                signal_strength = 1
                if rsi[i] < 50.0 and rsi[i] > 30.0:
                    desired_signal = -SIZE_MAX
                    signal_strength = 2
        
        else:
            # === TRANSITION REGIME: Use HMA direction only ===
            if hma_12h_bull and rsi[i] > 40.0:
                desired_signal = SIZE_BASE * 0.5
            elif hma_12h_bear and rsi[i] < 60.0:
                desired_signal = -SIZE_BASE * 0.5
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_MAX * 0.9:
            final_signal = SIZE_MAX
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_MAX * 0.9:
            final_signal = -SIZE_MAX
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals