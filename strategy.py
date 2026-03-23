#!/usr/bin/env python3
"""
Experiment #144: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Fisher Reversals

Hypothesis: Previous regime-switching strategies failed due to overly complex logic.
This uses SIMPLER adaptive trend following with proven reversal signals:

1) KAMA(21) - Kaufman Adaptive MA that adjusts to volatility (better than EMA in chop)
2) Fisher Transform(9) - catches reversals in bear rallies (proven 2022-2025)
3) 12h HMA(21) for macro trend bias - only trade with HTF direction
4) Choppiness(14) simplified: >50 = reduce size, <50 = full size
5) ATR(14) trailing stop at 2.5x - mandatory capital protection
6) Volume filter: only 1.2x avg (not 1.8x which kills trades)

Why this should work:
- KAMA adapts to market conditions automatically (no regime switch needed)
- Fisher Transform proven in crypto bear markets (catches 20-30% reversals)
- 12h HMA gives macro bias without over-filtering
- Simpler = more trades (target 35-50/year on 4h)
- Discrete sizes minimize fee churn

Position size: 0.25 base, 0.30 with confluence
Stoploss: 2.5*ATR trailing
Target: 35-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_hma_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts to market volatility.
    ER (Efficiency Ratio) determines smoothing constant.
    Better than EMA in choppy markets.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio: |change| / sum of absolute changes
    change = np.abs(close_s.diff(period))
    volatility = close_s.diff().abs().rolling(window=period, min_periods=period).sum()
    er = change / (volatility + 1e-10)
    er = er.fillna(0).values
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[period-1] = close[period-1]
    
    for i in range(period, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_fisher(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals at extreme values (-2 to +2 range).
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Normalize price within period range
    highest = hl2_s.rolling(window=period, min_periods=period).max()
    lowest = hl2_s.rolling(window=period, min_periods=period).min()
    range_val = highest - lowest
    range_val = np.maximum(range_val, 1e-10)
    
    normalized = (hl2 - lowest) / range_val * 2.0 - 1.0
    normalized = normalized.clip(-0.99, 0.99)
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
    fisher_prev = fisher.shift(1).fillna(0).values
    
    return fisher.values, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending.
    CHOP > 50 = choppy/range, CHOP < 50 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for macro trend
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_21 = calculate_kama(close, period=21)
    fisher, fisher_prev = calculate_fisher(high, low, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_21[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        
        # === HTF TREND BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === LOCAL TREND (KAMA slope) ===
        kama_slope = kama_21[i] - kama_21[i-5] if i >= 5 else 0
        kama_uptrend = kama_slope > 0
        kama_downtrend = kama_slope < 0
        
        # === FISHER REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long = (fisher[i] > -1.5) and (fisher_prev[i] <= -1.5)
        # Short: Fisher crosses below +1.5 from above
        fisher_short = (fisher[i] < 1.5) and (fisher_prev[i] >= 1.5)
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 50.0
        is_trending = chop_14[i] < 50.0
        
        # === VOLUME ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 1.2
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Fisher reversal + HTF bias + KAMA confirmation ---
        if fisher_long:
            # Must have HTF support (price above 12h HMA) OR strong KAMA uptrend
            if price_above_hma_12h or (kama_uptrend and close[i] > kama_21[i]):
                if volume_confirmed or not is_choppy:
                    new_signal = POSITION_SIZE_BASE
                    # Increase size if trending + volume
                    if is_trending and volume_ratio > 1.5:
                        new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY: Fisher reversal + HTF bias + KAMA confirmation ---
        if fisher_short:
            # Must have HTF support (price below 12h HMA) OR strong KAMA downtrend
            if price_below_hma_12h or (kama_downtrend and close[i] < kama_21[i]):
                if volume_confirmed or not is_choppy:
                    new_signal = -POSITION_SIZE_BASE
                    # Increase size if trending + volume
                    if is_trending and volume_ratio > 1.5:
                        new_signal = -POSITION_SIZE_MAX
        
        # --- TREND FOLLOWING: KAMA crossover with HTF confirmation ---
        # Long: price crosses above KAMA + above 12h HMA
        if close[i] > kama_21[i] and close[i-1] <= kama_21[i-1]:
            if price_above_hma_12h:
                if new_signal == 0.0:  # Don't override Fisher signal
                    new_signal = POSITION_SIZE_BASE
        
        # Short: price crosses below KAMA + below 12h HMA
        if close[i] < kama_21[i] and close[i-1] >= kama_21[i-1]:
            if price_below_hma_12h:
                if new_signal == 0.0:
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if no new signal and position still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if above KAMA
                if close[i] > kama_21[i]:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if below KAMA
                if close[i] < kama_21[i]:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if price crosses below KAMA significantly
        if in_position and position_side > 0:
            if close[i] < kama_21[i] * 0.995:  # 0.5% buffer
                new_signal = 0.0
        
        # Exit short if price crosses above KAMA significantly
        if in_position and position_side < 0:
            if close[i] > kama_21[i] * 1.005:  # 0.5% buffer
                new_signal = 0.0
        
        # === EXIT ON HTF REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_12h and chop_14[i] < 40.0:  # Strong downtrend
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_12h and chop_14[i] < 40.0:  # Strong uptrend
                new_signal = 0.0
        
        # === EXIT ON FISHER EXTREME (take profit) ===
        if in_position and position_side > 0 and fisher[i] > 2.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -2.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals