#!/usr/bin/env python3
"""
Experiment #695: 6h Primary + 1d/1w HTF — Fisher Transform + Regime + Volatility

Hypothesis: 6h timeframe offers unique balance between 4h noise and 12h lag.
Fisher Transform catches reversals better than RSI in bear/range markets (2022 crash, 2025 bear).
Combined with volatility regime (ATR ratio) and HTF bias for directional filter.

Key innovations:
1. Fisher Transform (period=9) - Gaussian normalization, catches extremes at -2.0/+2.0
2. Volatility regime: ATR(7)/ATR(21) ratio - >1.8 = vol spike (reversion), <0.8 = quiet (trend)
3. 1d/1w HMA bias - only trade with HTF direction for asymmetry
4. BB squeeze detection - BW percentile <20% = compression before expansion
5. ATR(14) trailing stop - 2.5x for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Entry conditions (LOOSE to ensure trades):
- LONG: Fisher<-1.5 OR (vol_spike + price<BB_lower) + HTF bull bias
- SHORT: Fisher>+1.5 OR (vol_spike + price>BB_upper) + HTF bear bias
- No strict ADX/CHOP filters (caused 0 trades in past experiments)

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_volregime_bb_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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

def calculate_fisher(close, period=9):
    """Ehlers Fisher Transform - normalizes price to Gaussian distribution"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate midprice
    midprice = (pd.Series(close).rolling(period, min_periods=period).max().values + 
                pd.Series(close).rolling(period, min_periods=period).min().values) / 2.0
    
    # Normalize to -1 to +1 range
    highest = pd.Series(close).rolling(period, min_periods=period).max().values
    lowest = pd.Series(close).rolling(period, min_periods=period).min().values
    
    range_val = highest - lowest
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    normalized = 2.0 * (close - lowest) / range_val - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    
    # Signal line (previous fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = np.nan
    
    return fisher, fisher_signal

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(period, min_periods=period).mean().values
    std = pd.Series(close).rolling(period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    
    return upper, lower, width

def calculate_vol_ratio(atr, short_period=7, long_period=21):
    """ATR ratio for volatility regime detection"""
    n = len(atr)
    if n < long_period:
        return np.full(n, np.nan)
    
    atr_short = pd.Series(atr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(atr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher, fisher_signal = calculate_fisher(close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_ratio = calculate_vol_ratio(atr, short_period=7, long_period=21)
    
    # BB width percentile for squeeze detection
    bb_width_pct = pd.Series(bb_width).rolling(60, min_periods=60).rank(pct=True).values
    
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
        
        if np.isnan(fisher[i]) or np.isnan(bb_upper[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY REGIME ===
        vol_spike = vol_ratio[i] > 1.6  # ATR short > 1.6x ATR long
        vol_quiet = vol_ratio[i] < 0.9  # Low volatility = trend environment
        bb_squeeze = bb_width_pct[i] < 0.25 if not np.isnan(bb_width_pct[i]) else False
        
        # === HTF BIAS (1d and 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === FISHER REVERSAL SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher cross above -1.5 (bullish reversal)
        fisher_bull_cross = False
        if not np.isnan(fisher_signal[i]):
            fisher_bull_cross = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        
        # Fisher cross below +1.5 (bearish reversal)
        fisher_bear_cross = False
        if not np.isnan(fisher_signal[i]):
            fisher_bear_cross = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # === BB REVERSION ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG entries - multiple pathways to ensure trades
        long_score = 0
        
        # Fisher reversal (strongest)
        if fisher_bull_cross:
            long_score += 2
        elif fisher_oversold:
            long_score += 1
        
        # Volatility spike + BB reversion
        if vol_spike and price_below_bb:
            long_score += 2
        elif price_below_bb and bb_squeeze:
            long_score += 1
        
        # HTF bias alignment
        if htf_1d_bull and htf_1w_bull:
            long_score += 1
        elif htf_1d_bull:
            long_score += 0.5
        
        if long_score >= 2.5:
            desired_signal = SIZE_STRONG
        elif long_score >= 1.5:
            desired_signal = SIZE_BASE
        
        # SHORT entries - multiple pathways
        short_score = 0
        
        # Fisher reversal (strongest)
        if fisher_bear_cross:
            short_score += 2
        elif fisher_overbought:
            short_score += 1
        
        # Volatility spike + BB reversion
        if vol_spike and price_above_bb:
            short_score += 2
        elif price_above_bb and bb_squeeze:
            short_score += 1
        
        # HTF bias alignment
        if htf_1d_bear and htf_1w_bear:
            short_score += 1
        elif htf_1d_bear:
            short_score += 0.5
        
        if short_score >= 2.5:
            desired_signal = -SIZE_STRONG
        elif short_score >= 1.5:
            desired_signal = -SIZE_BASE
        
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