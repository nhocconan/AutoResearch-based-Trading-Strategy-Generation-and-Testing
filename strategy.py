#!/usr/bin/env python3
"""
Experiment #684: 4h Primary + 12h HTF — Fisher Transform + KAMA + Choppiness

Hypothesis: Simplified regime detection with Fisher Transform entries will generate
more trades than complex CRSI+Donchian combinations while maintaining edge.

Key innovations:
1. Ehlers Fisher Transform (period=9) — superior reversal detection vs RSI
   Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. KAMA (Kaufman Adaptive MA) — adapts to volatility, less whipsaw than HMA
   Efficiency Ratio determines smoothing constant dynamically
3. Choppiness Index — simple binary regime (CHOP>55=range, CHOP<45=trend)
4. 12h HMA for macro bias — single HTF filter (not 1d+1w complexity)
5. ATR-based position sizing — reduce size when volatility spikes
6. LOOSE entry thresholds — Fisher > -1.8 (not -1.5) to ensure trade generation

Why this should work where others failed:
- Fisher Transform has better reversal timing than RSI/CRSI in research
- KAMA adapts smoothing based on market efficiency (less lag in trends)
- Single 12h HTF filter reduces complexity vs 1d+1w combinations
- Looser Fisher thresholds ensure 30+ trades/year on 4h timeframe
- ATR sizing reduces position when vol spikes (protects during crashes)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_chop_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Catches reversals better than RSI in bear/range markets.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5.
    We use -1.8/+1.8 for looser entries to ensure trade generation.
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    # Calculate typical price
    typical = (high + low + close) / 3
    
    # Normalize to -1 to +1 range
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest[i] = np.max(typical[i - period + 1:i + 1])
        lowest[i] = np.min(typical[i - period + 1:i + 1])
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    normalized = 2 * (typical - lowest) / range_val - 1
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher_raw = np.zeros(n)
    for i in range(period - 1, n):
        if not np.isnan(normalized[i]):
            fisher_raw[i] = 0.5 * np.log((1 + normalized[i]) / (1 - normalized[i] + 1e-10))
    
    # Smooth with EMA
    fisher = pd.Series(fisher_raw).ewm(span=3, min_periods=3, adjust=False).mean().values
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[:period] = np.nan
    
    return fisher, fisher_signal

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts smoothing based on market efficiency.
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    High ER = trending (use fast SC), Low ER = choppy (use slow SC)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period - 1, n):
        net_change = np.abs(close[i] - close[i - er_period + 1])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period + 1:i + 1])))
        if sum_changes > 1e-10:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    
    er = np.clip(er, 0, 1)
    
    # Calculate Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = sc ** 2  # Square for smoother adaptation
    
    # Initialize KAMA
    kama[er_period - 1] = close[er_period - 1]
    
    # Calculate KAMA
    for i in range(er_period, n):
        if not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = close[i]
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — identifies ranging vs trending markets.
    CHOP > 61.8 = choppy/ranging (mean-revert)
    CHOP < 38.2 = trending (trend-follow)
    We use 55/45 thresholds for smoother transitions.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high > lowest_low:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr1 = high[j] - low[j]
                tr2 = np.abs(high[j] - close[j - 1]) if j > 0 else tr1
                tr3 = np.abs(low[j] - close[j - 1]) if j > 0 else tr1
                atr_sum += max(tr1, tr2, tr3)
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 100
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_fisher_cross(fisher, fisher_signal):
    """Detect Fisher Transform crossovers."""
    n = len(fisher)
    cross_up = np.zeros(n, dtype=bool)
    cross_down = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        if not np.isnan(fisher[i]) and not np.isnan(fisher_signal[i]):
            if not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
                # Cross above -1.8 (bullish reversal)
                if fisher_signal[i-1] <= -1.8 and fisher[i] > -1.8:
                    cross_up[i] = True
                # Cross below +1.8 (bearish reversal)
                if fisher_signal[i-1] >= 1.8 and fisher[i] < 1.8:
                    cross_down[i] = True
    
    return cross_up, cross_down

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (4h) indicators
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, close, period=9)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate Fisher crossovers
    fisher_cross_up, fisher_cross_down = calculate_fisher_cross(fisher_4h, fisher_signal_4h)
    
    # Calculate and align HTF (12h) indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_4h[i]
        is_range_regime = chop_value > 55
        is_trend_regime = chop_value < 45
        
        # === 12H MACRO BIAS ===
        macro_bullish = close[i] > hma_12h_aligned[i]
        macro_bearish = close[i] < hma_12h_aligned[i]
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === FISHER SIGNALS (LOOSE thresholds) ===
        fisher_oversold = fisher_4h[i] < -1.5
        fisher_overbought = fisher_4h[i] > 1.5
        fisher_extreme_oversold = fisher_4h[i] < -2.0
        fisher_extreme_overbought = fisher_4h[i] > 2.0
        
        # === ATR-BASED POSITION SIZING ===
        # Reduce size when ATR spikes (vol > 1.5x recent average)
        atr_ratio = atr_4h[i] / (pd.Series(atr_4h).ewm(span=20, min_periods=20).mean().values[i] + 1e-10)
        vol_scaling = 1.0 if atr_ratio < 1.5 else max(0.5, 2.0 - atr_ratio)
        
        position_size = BASE_SIZE * vol_scaling
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (CHOP < 45) — Trend Follow ===
        if is_trend_regime:
            # Long: Macro bullish + KAMA bullish + Fisher oversold or cross up
            if macro_bullish and kama_bullish:
                if fisher_cross_up[i] or (fisher_oversold and fisher_4h[i] > fisher_signal_4h[i]):
                    desired_signal = position_size
            
            # Short: Macro bearish + KAMA bearish + Fisher overbought or cross down
            elif macro_bearish and kama_bearish:
                if fisher_cross_down[i] or (fisher_overbought and fisher_4h[i] < fisher_signal_4h[i]):
                    desired_signal = -position_size
        
        # === REGIME 2: RANGING (CHOP > 55) — Mean Reversion ===
        elif is_range_regime:
            # Long: Fisher extreme oversold + price below KAMA (oversold bounce)
            if fisher_extreme_oversold or (fisher_oversold and close[i] < kama_4h[i]):
                if fisher_4h[i] > fisher_signal_4h[i]:  # Fisher turning up
                    desired_signal = position_size * 0.7
            
            # Short: Fisher extreme overbought + price above KAMA (overbought fade)
            if fisher_extreme_overbought or (fisher_overbought and close[i] > kama_4h[i]):
                if fisher_4h[i] < fisher_signal_4h[i]:  # Fisher turning down
                    desired_signal = -position_size * 0.7
        
        # === REGIME 3: TRANSITION (45 <= CHOP <= 55) — Mixed ===
        else:
            # Use Fisher with KAMA bias only
            if fisher_extreme_oversold and kama_bullish:
                desired_signal = position_size * 0.5
            elif fisher_extreme_overbought and kama_bearish:
                desired_signal = -position_size * 0.5
            elif fisher_cross_up[i] and macro_bullish:
                desired_signal = position_size * 0.5
            elif fisher_cross_down[i] and macro_bearish:
                desired_signal = -position_size * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if macro_bullish and fisher_4h[i] < 2.0:
                    desired_signal = position_size
            elif position_side < 0:
                if macro_bearish and fisher_4h[i] > -2.0:
                    desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = position_size
        elif desired_signal < 0:
            desired_signal = -position_size
        
        # Round to discrete levels
        if abs(desired_signal) > 0:
            if desired_signal > 0:
                desired_signal = round(desired_signal * 4) / 4  # 0.25, 0.30, 0.35
                desired_signal = min(max(desired_signal, 0.20), 0.35)
            else:
                desired_signal = -round(abs(desired_signal) * 4) / 4
                desired_signal = max(min(desired_signal, -0.20), -0.35)
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
        
        signals[i] = desired_signal
    
    return signals