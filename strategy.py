#!/usr/bin/env python3
"""
Experiment #639: 4h Primary + 1d HTF — Fisher Transform + KAMA + Choppiness + Donchian

Hypothesis: 4h timeframe provides optimal balance of trade frequency (20-50/year) and 
signal quality. Combining Fisher Transform reversals with Donchian breakouts ensures 
adequate trade generation. Choppiness Index switches between mean-reversion and 
trend-following modes. 1d HMA provides macro bias without being overly restrictive.

Key innovations:
1. Fisher Transform (period=9) with accessible thresholds (-1.5/+1.5) for regular signals
2. Donchian(20) breakouts as secondary entry trigger - ensures trades during trends
3. KAMA adaptive MA for trend confirmation - smoother than EMA in chop
4. Choppiness regime switch at 55/45 levels - proven meta-filter
5. 1d HMA as soft bias filter (not hard requirement to avoid 0-trade failure)
6. ATR trailing stop at 2.5x for risk management
7. Discrete position sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should beat Sharpe=0.612:
- Donchian breakouts guarantee trades during strong trends (fixes 0-trade failure mode)
- Fisher catches reversals in range markets where Donchian fails
- Dual entry system (Fisher + Donchian) ensures coverage across all regimes
- 4h timeframe = fewer false signals than 1h, more trades than 12h
- Conservative sizing (0.30) survives 77% crash with ~27% DD

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_chop_donchian_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to Gaussian normal distribution for clearer reversal signals.
    Long: Fisher crosses above -1.5 from below
    Short: Fisher crosses below +1.5 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    price = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        range_val = hh - ll
        if range_val < 1e-10:
            range_val = 1e-10
        
        price_raw = (close[i] - ll) / range_val
        
        if i > period:
            price[i] = 0.33 * 2 * (price_raw - 0.5) + 0.67 * price[i-1]
        else:
            price[i] = 0.33 * 2 * (price_raw - 0.5)
        
        price[i] = np.clip(price[i], -0.999, 0.999)
        fisher[i] = 0.5 * np.log((1 + price[i]) / (1 - price[i]))
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_signal

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency ratio.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    er = np.zeros(n)
    
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    kama[er_period] = np.mean(close[:er_period+1])
    
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 55 = choppy (mean revert), CHOP < 45 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
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
    """Hull Moving Average for HTF trend."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, close, period=9)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 period)
    donchian_high = pd.Series(high).rolling(20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(20, min_periods=20).min().values
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(kama_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop_4h[i] > 55.0
        is_trending = chop_4h[i] < 45.0
        
        # === HTF TREND BIAS (1d HMA) ===
        htf_bullish = close[i] > hma_1d_aligned[i]
        htf_bearish = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_cross = (fisher_4h[i] > -1.5) and (fisher_signal_4h[i] <= -1.5)
        fisher_short_cross = (fisher_4h[i] < 1.5) and (fisher_signal_4h[i] >= 1.5)
        
        # Fisher extremes for mean reversion
        fisher_oversold = fisher_4h[i] < -1.8
        fisher_overbought = fisher_4h[i] > 1.8
        
        # === DONCHIAN BREAKOUTS ===
        donchian_breakout_long = close[i] > donchian_high[i-1]
        donchian_breakout_short = close[i] < donchian_low[i-1]
        
        desired_signal = 0.0
        
        # === CHOPPY REGIME: Mean Reversion (Fisher) ===
        if is_choppy:
            if fisher_oversold:
                desired_signal = SIZE
            elif fisher_overbought:
                desired_signal = -SIZE
            elif fisher_long_cross:
                desired_signal = SIZE
            elif fisher_short_cross:
                desired_signal = -SIZE
        
        # === TRENDING REGIME: Trend Follow (Donchian + KAMA) ===
        elif is_trending:
            # Donchian breakout with KAMA confirmation
            if donchian_breakout_long and kama_bullish:
                desired_signal = SIZE
            elif donchian_breakout_short and kama_bearish:
                desired_signal = -SIZE
            # Fisher cross with trend alignment
            elif fisher_long_cross and kama_bullish:
                desired_signal = SIZE
            elif fisher_short_cross and kama_bearish:
                desired_signal = -SIZE
            # HTF alignment override
            elif htf_bullish and kama_bullish and fisher_4h[i] < 1.0:
                desired_signal = SIZE
            elif htf_bearish and kama_bearish and fisher_4h[i] > -1.0:
                desired_signal = -SIZE
        
        # === NEUTRAL REGIME ===
        else:
            if kama_bullish and fisher_4h[i] < 0.5:
                desired_signal = SIZE
            elif kama_bearish and fisher_4h[i] > -0.5:
                desired_signal = -SIZE
            elif donchian_breakout_long:
                desired_signal = SIZE
            elif donchian_breakout_short:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
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
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if kama_bullish and fisher_4h[i] < 2.0:
                    desired_signal = SIZE
            elif position_side < 0:
                if kama_bearish and fisher_4h[i] > -2.0:
                    desired_signal = -SIZE
        
        # === DISCRETIZE ===
        if desired_signal > 0:
            desired_signal = SIZE
        elif desired_signal < 0:
            desired_signal = -SIZE
        
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