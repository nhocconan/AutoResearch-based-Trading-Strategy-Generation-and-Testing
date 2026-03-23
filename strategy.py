#!/usr/bin/env python3
"""
Experiment #252: 12h Primary + 1d/1w HTF — Dual Regime Adaptive Strategy

Hypothesis: Based on proven patterns from experiment history:
- Choppiness Index regime filter worked for ETH (Sharpe +0.923 in research)
- HMA crossover + RSI filter worked for SOL (+0.879)
- Donchian breakout + HMA trend + RSI + ATR worked for SOL (+0.782)

KEY INSIGHT: Single-regime strategies fail because BTC/ETH spend 60%+ time ranging.
Need DUAL REGIME: mean-revert in chop (CHOP>55), trend-follow otherwise (CHOP<45).

CRITICAL FIX FROM FAILURES (#240, #242, #248, #250 = 0 trades):
- Loosen RSI thresholds: 25/75 for mean reversion (not 15/85 CRSI)
- Loosen CHOP thresholds: 45/55 neutral zone (not 38.2/61.8 extreme)
- Reduce confluence: 2-3 filters max, not 5+
- Position size 0.25-0.30 for 12h volatility

TARGET: 25-45 trades/year on 12h, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_hma_rsi_chop_1d_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 55 = ranging market (mean reversion)
    CHOP < 45 = trending market (trend following)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
    
    return chop.clip(0, 100).fillna(50.0).values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_mult * std)
    lower = sma - (std_mult * std)
    return sma.values, upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_sma, bb_upper, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # Calculate 1d HMA for macro trend (aligned properly with shift(1))
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12h TREND (HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = ranging (mean reversion)
        # CHOP < 45 = trending (trend follow)
        # 45-55 = neutral (use trend logic)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === BOLLINGER POSITION ===
        bb_position = (close[i] - bb_sma[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        near_bb_lower = close[i] < bb_lower[i] * 1.005
        near_bb_upper = close[i] > bb_upper[i] * 0.995
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === MEAN REVERSION REGIME (choppy market) ===
        if is_choppy:
            # Long: RSI oversold + near lower BB
            if rsi_14[i] < 30.0 and near_bb_lower:
                desired_signal = POSITION_SIZE_FULL
            # Short: RSI overbought + near upper BB
            elif rsi_14[i] > 70.0 and near_bb_upper:
                desired_signal = -POSITION_SIZE_FULL
        
        # === TREND FOLLOWING REGIME (trending market) ===
        elif is_trending:
            # Long: 1d bullish + 12h bullish + RSI pullback (40-55)
            if price_above_hma_1d and hma_bullish and 38.0 <= rsi_14[i] <= 58.0:
                desired_signal = POSITION_SIZE_FULL
            # Short: 1d bearish + 12h bearish + RSI pullback (42-62)
            elif price_below_hma_1d and hma_bearish and 42.0 <= rsi_14[i] <= 62.0:
                desired_signal = -POSITION_SIZE_FULL
        
        # === NEUTRAL REGIME (use trend bias only) ===
        else:
            # Long: 1d bullish + 12h bullish + RSI not overbought
            if price_above_hma_1d and hma_bullish and rsi_14[i] < 65.0:
                desired_signal = POSITION_SIZE_FULL
            # Short: 1d bearish + 12h bearish + RSI not oversold
            elif price_below_hma_1d and hma_bearish and rsi_14[i] > 35.0:
                desired_signal = -POSITION_SIZE_FULL
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_bearish:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if hma_bullish and price_above_hma_1d:
                    desired_signal = POSITION_SIZE_HALF
            elif position_side < 0:
                if hma_bearish and price_below_hma_1d:
                    desired_signal = -POSITION_SIZE_HALF
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
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