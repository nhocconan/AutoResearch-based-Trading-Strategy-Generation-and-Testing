#!/usr/bin/env python3
"""
Experiment #309: 4h Primary + 1d HTF — Fisher-KAMA Regime Strategy

Hypothesis: Current best (#301 Sharpe=0.612) uses CRSI+Donchian. This tries a DIFFERENT combo:
- Ehlers Fisher Transform (period=9): Cleaner reversal signals than RSI, less whipsaw
- KAMA (Kaufman Adaptive MA): Adapts to market noise better than HMA/EMA
- Choppiness Index(14): Regime detection (proven in #299)
- 1d KAMA for macro bias (instead of HMA)

KEY DIFFERENCES from #299 and failed experiments:
1. Fisher Transform instead of CRSI/RSI - catches reversals earlier with less noise
2. KAMA instead of HMA - adapts efficiency ratio to volatility (better in chop)
3. Two-tier exit: 50% at 1.5R profit, rest at ATR trail (locks gains)
4. Position size: 0.25 (conservative for 4h, allows room for pyramiding)
5. Looser Fisher thresholds (-1.2/+1.2 vs -1.5/+1.5) for MORE trades

TARGET: 30-50 trades/year, Sharpe > 0.7 on ALL symbols (beat #301's 0.612)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_regime_1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s.diff(er_period))
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]) or i < er_period:
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for cleaner reversal signals.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X is normalized price
    """
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate typical price range
    hl2 = (high_s + low_s) / 2
    
    # Normalize price within range
    highest = hl2.rolling(window=period, min_periods=period).max()
    lowest = hl2.rolling(window=period, min_periods=period).min()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        x = (hl2 - lowest) / (highest - lowest + 1e-10)
    
    # Clamp to avoid division by zero in Fisher formula
    x = x.clip(0.001, 0.999)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    
    # Calculate and align 1d KAMA for macro bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Conservative for 4h
    
    # Position tracking for stoploss and take-profit
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    partial_exit_done = False  # Track if we took 50% profit
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(kama_4h[i]):
            signals[i] = 0.0
            continue
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.2 = bullish reversal
        # Fisher crosses below +1.2 = bearish reversal
        fisher_bullish = fisher[i] > -1.2 and fisher_signal[i] <= -1.2
        fisher_bearish = fisher[i] < 1.2 and fisher_signal[i] >= 1.2
        
        # Fisher extreme levels for mean reversion in chop
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # === KAMA TREND FILTER ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Fisher Mean Reversion at extremes
            # Long when Fisher < -1.5 + above 1d KAMA (bullish bias)
            if fisher_oversold and price_above_kama_1d:
                desired_signal = POSITION_SIZE
            # Short when Fisher > 1.5 + below 1d KAMA (bearish bias)
            elif fisher_overbought and price_below_kama_1d:
                desired_signal = -POSITION_SIZE
        
        else:  # is_trending or neutral (45-55)
            # TREND REGIME: Fisher crossover + KAMA confirmation
            # LONG: Fisher crosses -1.2 + price > 4h KAMA + bullish 1d bias
            if fisher_bullish and kama_bullish and price_above_kama_1d:
                desired_signal = POSITION_SIZE
            # SHORT: Fisher crosses +1.2 + price < 4h KAMA + bearish 1d bias
            elif fisher_bearish and kama_bearish and price_below_kama_1d:
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
        
        # === TAKE PROFIT (50% at 1.5R, rest at trail) ===
        if in_position and not partial_exit_done:
            if position_side > 0:
                profit_r = (close[i] - entry_price) / atr_14[i]
                if profit_r >= 1.5:
                    # Take 50% profit (reduce signal to half)
                    desired_signal = POSITION_SIZE / 2
                    partial_exit_done = True
            elif position_side < 0:
                profit_r = (entry_price - close[i]) / atr_14[i]
                if profit_r >= 1.5:
                    desired_signal = -POSITION_SIZE / 2
                    partial_exit_done = True
        
        # === MACRO BIAS REVERSAL EXIT ===
        # Exit long if price crosses below 1d KAMA
        if in_position and position_side > 0 and price_below_kama_1d:
            desired_signal = 0.0
        
        # Exit short if price crosses above 1d KAMA
        if in_position and position_side < 0 and price_above_kama_1d:
            desired_signal = 0.0
        
        # === FISHER REVERSAL EXIT ===
        # Exit long if Fisher goes overbought (>1.5)
        if in_position and position_side > 0 and fisher[i] > 1.5:
            desired_signal = 0.0
        
        # Exit short if Fisher goes oversold (<-1.5)
        if in_position and position_side < 0 and fisher[i] < -1.5:
            desired_signal = 0.0
        
        # === HOLD LOGIC (maintain position unless exit trigger) ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Only hold if macro bias still supports position
            if position_side > 0 and price_above_kama_1d:
                if partial_exit_done:
                    desired_signal = POSITION_SIZE / 2  # Hold reduced position
                else:
                    desired_signal = POSITION_SIZE  # Hold full position
            elif position_side < 0 and price_below_kama_1d:
                if partial_exit_done:
                    desired_signal = -POSITION_SIZE / 2  # Hold reduced position
                else:
                    desired_signal = -POSITION_SIZE  # Hold full position
        
        # === UPDATE POSITION TRACKING ===
        # Only reset tracking when fully exiting (signal goes to 0)
        if desired_signal == 0.0:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
                partial_exit_done = False
        elif not in_position:
            # New position
            in_position = True
            position_side = int(np.sign(desired_signal))
            entry_price = close[i]
            if position_side > 0:
                highest_since_entry = close[i]
                lowest_since_entry = float('inf')
            else:
                highest_since_entry = 0.0
                lowest_since_entry = close[i]
            partial_exit_done = False
        elif np.sign(desired_signal) != position_side:
            # Position reversal
            position_side = int(np.sign(desired_signal))
            entry_price = close[i]
            if position_side > 0:
                highest_since_entry = close[i]
                lowest_since_entry = float('inf')
            else:
                highest_since_entry = 0.0
                lowest_since_entry = close[i]
            partial_exit_done = False
        
        signals[i] = desired_signal
    
    return signals