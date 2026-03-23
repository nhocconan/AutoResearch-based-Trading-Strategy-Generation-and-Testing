#!/usr/bin/env python3
"""
Experiment #369: 4h Primary + 1d HTF — Fisher Transform + Regime Adaptive

Hypothesis: Fisher Transform excels at catching reversals in bear market rallies,
while Choppiness Index determines mean-revert vs trend-follow mode. 1d HMA provides
macro trend bias. This should work through 2022 crash AND 2025 bear/range.

KEY DIFFERENCES from failed experiments:
1. Fisher Transform instead of Connors RSI (better for reversals in crypto)
2. Relaxed Fisher thresholds (-1.0/+1.0 instead of -1.5/+1.5) to ensure trades
3. RSI confirmation for additional entry triggers (more trade frequency)
4. Asymmetric logic: easier short entries in bear, selective longs
5. Remove BB requirement for trend regime (too restrictive)

TARGET: 30-60 trades/year on 4h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_regime_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((price - lowest_low) / (highest_high - lowest_low) - 0.5)
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            x = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5)
            x = np.clip(x, -0.99, 0.99)
            fisher[i] = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
            trigger[i] = fisher[i-1] if i > 0 else fisher[i]
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0
            trigger[i] = fisher[i]
    
    return fisher, trigger

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
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 4h (target 30-60 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range regime (mean revert)
        is_trending = chop[i] < 45.0  # Trend regime (breakout/follow)
        is_neutral = not is_choppy and not is_trending  # 45-55 neutral zone
        
        # === FISHER TRANSFORM SIGNALS (relaxed thresholds) ===
        # Long: Fisher crosses above -1.0 (oversold reversal)
        # Short: Fisher crosses below +1.0 (overbought reversal)
        fisher_long_cross = fisher[i] > -1.0 and fisher_trigger[i-1] <= -1.0
        fisher_short_cross = fisher[i] < 1.0 and fisher_trigger[i-1] >= 1.0
        
        # === RSI EXTREMES (additional entry trigger) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === BB POSITION ===
        at_bb_lower = close[i] < bb_lower[i]
        at_bb_upper = close[i] > bb_upper[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Mean reversion at BB bounds + Fisher/RSI confirmation
            # Long: BB lower + (Fisher long OR RSI oversold)
            # Short: BB upper + (Fisher short OR RSI overbought)
            
            long_cond = at_bb_lower and (fisher_long_cross or rsi_oversold)
            short_cond = at_bb_upper and (fisher_short_cross or rsi_overbought)
            
            if long_cond:
                desired_signal = BASE_SIZE
            elif short_cond:
                desired_signal = -BASE_SIZE
        
        elif is_trending:
            # TREND REGIME: Follow 1d HMA direction with Fisher/RSI entry
            # Long: Price > 1d HMA + (Fisher long OR RSI oversold pullback)
            # Short: Price < 1d HMA + (Fisher short OR RSI overbought rally)
            
            long_cond = price_above_hma_1d and (fisher_long_cross or rsi_oversold)
            short_cond = price_below_hma_1d and (fisher_short_cross or rsi_overbought)
            
            if long_cond:
                desired_signal = BASE_SIZE
            elif short_cond:
                desired_signal = -BASE_SIZE
        
        elif is_neutral:
            # NEUTRAL REGIME: Only take strong Fisher signals with 1d bias
            # More selective to avoid whipsaw
            if price_above_hma_1d and fisher_long_cross:
                desired_signal = BASE_SIZE
            elif price_below_hma_1d and fisher_short_cross:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === FISHER EXIT (reversal complete) ===
        if in_position and position_side > 0 and fisher[i] > 1.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.0:
            desired_signal = 0.0
        
        # === RSI EXIT ===
        if in_position and position_side > 0 and rsi_14[i] > 70:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if Fisher hasn't reversed and bias intact
                if fisher[i] < 1.0 and (is_choppy or price_above_hma_1d):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if Fisher hasn't reversed and bias intact
                if fisher[i] > -1.0 and (is_choppy or price_below_hma_1d):
                    desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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