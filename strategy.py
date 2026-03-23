#!/usr/bin/env python3
"""
Experiment #314: 4h Primary + 1d HTF — Fisher Transform + KAMA Adaptive Trend

Hypothesis: Previous regime strategies over-filtered entries. This combines:
- Ehlers Fisher Transform (period=9): Superior reversal detection in bear markets
- KAMA (Kaufman Adaptive MA): Adapts to volatility, reduces whipsaw vs EMA/HMA
- Choppiness Index: Regime switch (mean revert vs trend follow)
- 1d HMA(21): Macro bias filter (proven in #312, #313)
- ATR(14) 2.5x trailing stop: Protects from 2022-style crashes

KEY DIFFERENCES from failed experiments:
- Fisher Transform catches reversals better than RSI/CRSI in bear markets
- KAMA adapts to volatility (ER-based smoothing) vs fixed-period HMA
- Simpler entry logic: fewer confluence requirements = more trades
- No session/volume filters that killed trades in #305, #308, #310

TARGET: 25-40 trades/year on 4h, Sharpe > 0.6 on ALL symbols (BTC, ETH, SOL)
Position size: 0.30 (conservative for 4h, limits DD to ~30% on 77% crash)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_chop_1d_hma_atr_v1"
timeframe = "4h"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on Efficiency Ratio (ER).
    ER = |price change| / sum of absolute price changes
    High ER = trending (fast smoothing), Low ER = choppy (slow smoothing)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    price_change = np.abs(close_s.diff(er_period))
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Normalize price within lookback period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = fisher[i-1] if i > 0 else 0
            fisher_signal[i] = fisher[i]
            continue
        
        # Normalize to -1 to +1 range
        norm = 2 * ((high[i] + low[i]) / 2 - lowest) / (highest - lowest) - 1
        norm = np.clip(norm, -0.999, 0.999)  # Prevent log domain errors
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1 + norm) / (1 - norm))
        
        # Signal line (1-period lag)
        fisher_signal[i] = fisher[i-1] if i > 0 else 0
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crossings for cleaner signals
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(kama[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range (mean revert), CHOP < 45 = trend (trend follow)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Track Fisher crossing
        fisher_crossed_up = (prev_fisher < -1.5 and fisher[i] >= -1.5)
        fisher_crossed_down = (prev_fisher > 1.5 and fisher[i] <= 1.5)
        
        if is_choppy:
            # RANGE REGIME: Fisher Transform Mean Reversion
            # Long: Fisher crosses above -1.5 (oversold) + bullish macro
            if fisher_crossed_up and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            # Short: Fisher crosses below +1.5 (overbought) + bearish macro
            elif fisher_crossed_down and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        
        elif is_trending:
            # TREND REGIME: KAMA crossover with RSI confirmation
            # Long: Price > KAMA + RSI > 50 + bullish macro
            if close[i] > kama[i] and rsi_14[i] > 50.0 and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            # Short: Price < KAMA + RSI < 50 + bearish macro
            elif close[i] < kama[i] and rsi_14[i] < 50.0 and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        
        else:
            # NEUTRAL REGIME: Use Fisher for entries (more sensitive)
            # More lenient Fisher thresholds in neutral
            fisher_crossed_up_neutral = (prev_fisher < -1.0 and fisher[i] >= -1.0)
            fisher_crossed_down_neutral = (prev_fisher > 1.0 and fisher[i] <= 1.0)
            
            if fisher_crossed_up_neutral and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            elif fisher_crossed_down_neutral and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        
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
        
        # === MACRO BIAS REVERSAL EXIT ===
        # Exit long if price crosses below 1d HMA
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        # Exit short if price crosses above 1d HMA
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === FISHER EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and fisher[i] > 1.5:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.5:
            desired_signal = 0.0
        
        # === HOLD LOGIC (maintain position unless exit trigger) ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Only hold if macro bias still supports position
            if position_side > 0 and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_hma_1d:
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
        
        # Update previous Fisher for next iteration
        prev_fisher = fisher[i]
        
        signals[i] = desired_signal
    
    return signals