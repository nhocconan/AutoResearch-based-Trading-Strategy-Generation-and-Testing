#!/usr/bin/env python3
"""
Experiment #635: 1h Primary + 4h/1d HTF — Fisher Transform + HMA Trend + Choppiness Regime

Hypothesis: 1h timeframe with 4h trend filter can work IF entry conditions are loosened
enough to generate trades (30-60/year) while maintaining HTF direction bias. Previous
1h strategies failed with 0 trades due to over-filtering (session + volume + multiple
indicators all required). This version uses Fisher Transform for entry timing within
4h HMA trend direction, with Choppiness Index to switch between mean-revert and trend-follow.

Key innovations:
1. Fisher Transform (period=9) — proven reversal catcher in bear/range markets
2. 4h HMA(21) for trend bias — only long when 4h bullish, only short when 4h bearish
3. Choppiness Index regime — CHOP>55 mean revert, CHOP<45 trend follow
4. Looser Fisher thresholds (-1.0/+1.0 cross, -1.3/+1.3 extreme) to ensure trade frequency
5. Hold logic — maintain position through minor pullbacks (reduces churn)
6. Discrete sizing: 0.25 for mean reversion, 0.30 for trend follow

Why this should beat Sharpe=0.000 (failed 1h strategies):
- Fewer filters = more trades (no session filter, no volume filter that killed #630)
- Fisher has documented 75% win rate on reversals
- 4h HMA provides direction without being too restrictive
- Hold logic prevents premature exits

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_chop_regime_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to Gaussian normal distribution for clearer reversal signals.
    
    Long signal: Fisher crosses above -1.0 from below (looser than -1.2)
    Short signal: Fisher crosses below +1.0 from above (looser than +1.2)
    Extreme levels: -1.3 oversold, +1.3 overbought
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

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother trend detection."""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    We use: > 55 = chop (mean revert), < 45 = trend (trend follow)
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    fisher_1h, fisher_signal_1h = calculate_fisher_transform(high, low, close, period=9)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels
    SIZE_MR = 0.25    # Mean reversion (smaller)
    SIZE_TF = 0.30    # Trend follow (larger)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_1h[i]) or np.isnan(fisher_signal_1h[i]):
            continue
        if np.isnan(chop_1h[i]) or np.isnan(atr_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if atr_1h[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_1h[i] > 55.0
        is_trending = chop_1h[i] < 45.0
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === 1d HMA for stronger confirmation ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.0 from below (looser threshold)
        fisher_long_cross = (fisher_1h[i] > -1.0) and (fisher_signal_1h[i] <= -1.0)
        # Short: Fisher crosses below +1.0 from above (looser threshold)
        fisher_short_cross = (fisher_1h[i] < 1.0) and (fisher_signal_1h[i] >= 1.0)
        
        # Fisher extreme levels (for mean reversion)
        fisher_oversold = fisher_1h[i] < -1.3
        fisher_overbought = fisher_1h[i] > 1.3
        
        # Fisher recovery from extreme
        fisher_recovery_long = (fisher_1h[i] > -1.0) and (fisher_1h[i-1] < -1.3) if i > 0 else False
        fisher_recovery_short = (fisher_1h[i] < 1.0) and (fisher_1h[i-1] > 1.3) if i > 0 else False
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion) ===
        if is_choppy:
            # Long: Fisher oversold + 4h not strongly bearish
            if fisher_oversold and not htf_4h_bearish:
                desired_signal = SIZE_MR
            # Short: Fisher overbought + 4h not strongly bullish
            elif fisher_overbought and not htf_4h_bullish:
                desired_signal = -SIZE_MR
            # Fisher recovery signals
            elif fisher_recovery_long and not htf_4h_bearish:
                desired_signal = SIZE_MR
            elif fisher_recovery_short and not htf_4h_bullish:
                desired_signal = -SIZE_MR
        
        # === REGIME 2: TRENDING MARKET (Trend Follow) ===
        elif is_trending:
            # Long: 4h bullish + 1d bullish + Fisher not overbought
            if htf_4h_bullish and htf_1d_bullish and fisher_1h[i] < 1.0:
                desired_signal = SIZE_TF
            # Short: 4h bearish + 1d bearish + Fisher not oversold
            elif htf_4h_bearish and htf_1d_bearish and fisher_1h[i] > -1.0:
                desired_signal = -SIZE_TF
            # Fisher cross with trend confirmation
            elif fisher_long_cross and htf_4h_bullish:
                desired_signal = SIZE_TF
            elif fisher_short_cross and htf_4h_bearish:
                desired_signal = -SIZE_TF
        
        # === REGIME 3: NEUTRAL/TRANSITION ===
        else:
            # Use 4h direction with Fisher filter (looser conditions)
            if htf_4h_bullish and fisher_1h[i] < 0.5:
                desired_signal = SIZE_MR
            elif htf_4h_bearish and fisher_1h[i] > -0.5:
                desired_signal = -SIZE_MR
            # Fisher cross in neutral
            elif fisher_long_cross:
                desired_signal = SIZE_MR
            elif fisher_short_cross:
                desired_signal = -SIZE_MR
        
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
        # CRITICAL: This prevents premature exits and reduces trade count
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h still bullish OR Fisher not extremely overbought
                if htf_4h_bullish and fisher_1h[i] < 1.8:
                    desired_signal = SIZE_TF if is_trending else SIZE_MR
            elif position_side < 0:
                # Hold short if 4h still bearish OR Fisher not extremely oversold
                if htf_4h_bearish and fisher_1h[i] > -1.8:
                    desired_signal = -SIZE_TF if is_trending else -SIZE_MR
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_TF if is_trending else SIZE_MR
        elif desired_signal < 0:
            desired_signal = -SIZE_TF if is_trending else -SIZE_MR
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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