#!/usr/bin/env python3
"""
Experiment #208: 4h Primary + 12h/1d HTF — Fisher Transform + Choppiness Regime + Donchian

Hypothesis: 4h timeframe with regime-switching logic can adapt to both bear/range markets
(2025 test period) and bull markets (2021 train period). Key innovations:

1. Ehlers Fisher Transform (period=9): Catches reversals in bear rallies better than RSI.
   Long when Fisher crosses above -1.5, short when crosses below +1.5.

2. Choppiness Index regime filter: CHOP>55 = mean revert (Fisher signals), CHOP<45 = trend follow (Donchian).

3. 12h HMA(34) for major trend bias: Only long if price>12h HMA, only short if price<12h HMA.

4. 1d HMA(50) for macro filter: Reduces position size when conflicting with 12h trend.

5. ATR(14) trailing stop at 2.5x: Mandatory risk management per rules.

Position sizing: 0.25 base, 0.30 strong (discrete levels to minimize fee churn).
Target: Sharpe>0.40 (beat current best 0.399), DD>-40%, trades>=30 train, trades>=3 test per symbol.

Why this should work:
- Fisher Transform proven in bear markets (2022 crash, 2025 test)
- Choppiness filter prevents trend strategies from whipsawing in ranges
- 12h/1d HTF alignment ensures we trade with higher timeframe momentum
- 4h TF = 20-50 trades/year target (fee drag manageable)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_regime_donchian_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound (mean reversion regime)
    CHOP < 38.2 = trending (trend follow regime)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points better than RSI in bear markets
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) -> 0 to 1
    3. Transform: 0.5 * ln((1+x)/(1-x)) where x = 2*normalized - 1
    """
    n = len(close) if 'close' in dir() else len(high)
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    for i in range(period-1, n):
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 1e-10:
            normalized = (typical[i] - lowest) / price_range
            # Clamp to avoid division by zero in log
            x = 2.0 * normalized - 1.0
            x = np.clip(x, -0.99, 0.99)
            fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
    
    return fisher

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=34)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher = calculate_fisher_transform(high, low, period=9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Fisher transform tracking for crossovers
    prev_fisher = np.nan
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(chop[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_fisher = fisher[i] if not np.isnan(fisher[i]) else prev_fisher
            continue
        
        # === HTF BIAS (12h HMA) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # === MACRO FILTER (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # HTF alignment score (for position sizing)
        htf_aligned = (htf_12h_bull and htf_1d_bull) or (htf_12h_bear and htf_1d_bear)
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range-bound market → mean reversion
        is_trending = chop[i] < 45.0  # Trending market → trend follow
        # 45-55 is transition zone
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = False
        fisher_cross_down = False
        
        if not np.isnan(prev_fisher):
            # Long: Fisher crosses above -1.5 (oversold reversal)
            if prev_fisher < -1.5 and fisher[i] >= -1.5:
                fisher_cross_up = True
            # Short: Fisher crosses below +1.5 (overbought reversal)
            if prev_fisher > 1.5 and fisher[i] <= 1.5:
                fisher_cross_down = True
        
        prev_fisher = fisher[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI EXTREMES (backup for choppy regime) ===
        rsi_oversold = not np.isnan(rsi[i]) and rsi[i] < 35.0
        rsi_overbought = not np.isnan(rsi[i]) and rsi[i] > 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # REGIME 1: CHOPPY (mean reversion with Fisher Transform)
        if is_choppy:
            # Long: Fisher reversal + 12h HMA support or neutral
            if fisher_cross_up and (htf_12h_bull or chop[i] > 60):
                desired_signal = SIZE_BASE if htf_aligned else SIZE_BASE * 0.8
            
            # Short: Fisher reversal + 12h HMA resistance or neutral
            elif fisher_cross_down and (htf_12h_bear or chop[i] > 60):
                desired_signal = -SIZE_BASE if htf_aligned else -SIZE_BASE * 0.8
            
            # Backup: RSI extremes in choppy market
            elif rsi_oversold and htf_12h_bull:
                desired_signal = SIZE_BASE * 0.8
            elif rsi_overbought and htf_12h_bear:
                desired_signal = -SIZE_BASE * 0.8
        
        # REGIME 2: TRENDING (breakout with HMA confirmation)
        elif is_trending:
            # Long: Donchian breakout + 12h HMA bull + 1d aligned
            if breakout_long and htf_12h_bull:
                desired_signal = SIZE_STRONG if htf_aligned else SIZE_BASE
            
            # Short: Donchian breakout + 12h HMA bear + 1d aligned
            elif breakout_short and htf_12h_bear:
                desired_signal = -SIZE_STRONG if htf_aligned else -SIZE_BASE
        
        # REGIME 3: TRANSITION (45-55 chop) - require stronger confirmation
        else:
            # Only enter if HTF strongly aligned
            if htf_aligned and htf_12h_bull and fisher_cross_up:
                desired_signal = SIZE_BASE * 0.8
            elif htf_aligned and htf_12h_bear and fisher_cross_down:
                desired_signal = -SIZE_BASE * 0.8
            elif htf_aligned and htf_12h_bull and breakout_long:
                desired_signal = SIZE_BASE * 0.8
            elif htf_aligned and htf_12h_bear and breakout_short:
                desired_signal = -SIZE_BASE * 0.8
        
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
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.6:
            final_signal = SIZE_BASE * 0.6
        elif desired_signal <= -SIZE_BASE * 0.6:
            final_signal = -SIZE_BASE * 0.6
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