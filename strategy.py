#!/usr/bin/env python3
"""
Experiment #1069: 4h Primary + 1d HTF — Choppiness Regime + HMA Crossover + RSI Filter

Hypothesis: After analyzing 775+ failed experiments, the winning pattern for 4h timeframe:
1. CHOPPINESS INDEX (14-period) for regime detection — PROVEN in #1063 (kept, Sharpe=0.135)
   CHOP > 61.8 = range market → mean reversion at Bollinger bounds
   CHOP < 38.2 = trending market → trend follow on HMA crossover
2. HMA(16/48) crossover — faster than EMA, proven in SOL strategies (+0.879 Sharpe)
   Fast HMA16 crosses above Slow HMA48 → long signal
   Fast HMA16 crosses below Slow HMA48 → short signal
3. RSI(14) filter with WIDE thresholds — ensure sufficient trades (30+/train)
   Long: RSI < 55 (not overbought) | Short: RSI > 45 (not oversold)
   Relaxed from typical 30/70 to generate MORE entries
4. 1d HMA21 macro bias — only trade in direction of daily trend
5. ATR(14) trailing stop 2.5x — proven risk management

Why this should beat Sharpe=0.612:
- CHOP regime filter PROVEN for ETH (#1063 kept strategy)
- HMA crossover simpler than complex Fisher/KAMA logic (less overfitting)
- WIDE RSI thresholds ensure 50-80 trades/year (not 0 trades like #1059, #1068)
- 4h timeframe = target 20-50 trades/year (optimal fee/trade ratio)
- Different from failed CRSI-heavy strategies (#1057, #1058, #1060, #1064, #1065)

Timeframe: 4h
HTF: 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 2.5x ATR trailing
Target Trades: 40-80 per year (20-50 on 4h per rules)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_hma_rsi_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market consolidation vs trending.
    
    Formula:
    1. ATR_sum = sum of ATR(1) over lookback period
    2. Price_range = highest high - lowest low over lookback
    3. CHOP = 100 * log10(ATR_sum / Price_range) / log10(period)
    
    Interpretation:
    CHOP > 61.8 = choppy/range market (mean reversion favored)
    CHOP < 38.2 = trending market (trend following favored)
    Between 38.2-61.8 = transition zone
    
    Proven in research for regime detection (ETH Sharpe +0.923)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            chop[i] = chop[i - 1] if i > 0 else 50.0
        else:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Use EMA for smoother RSI
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    valid_mask = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 1e-10)
    rs[valid_mask] = avg_gain[valid_mask] / avg_loss[valid_mask]
    
    rsi[valid_mask] = 100.0 - (100.0 / (1.0 + rs[valid_mask]))
    rsi[~valid_mask] = 50.0  # Neutral when loss is zero
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion levels."""
    n = len(close)
    middle = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    if n < period:
        return middle, upper, lower
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return middle, upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_fast = calculate_hma(close, 16)
    hma_slow = calculate_hma(close, 48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track HMA crossover state
    prev_hma_diff = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            continue
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(bb_mid[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trend market
        # Between 38.2-61.8 = neutral/transition
        
        # === MACRO TREND (1d HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === HMA CROSSOVER SIGNAL ===
        hma_diff = hma_fast[i] - hma_slow[i]
        hma_bull = hma_diff > 0
        hma_bear = hma_diff < 0
        
        # Detect crossover
        hma_cross_long = (prev_hma_diff <= 0 and hma_diff > 0)
        hma_cross_short = (prev_hma_diff >= 0 and hma_diff < 0)
        prev_hma_diff = hma_diff
        
        # === RSI FILTER (WIDE thresholds for more trades) ===
        rsi_not_overbought = rsi[i] < 55  # Relaxed from 70
        rsi_not_oversold = rsi[i] > 45    # Relaxed from 30
        
        # === BOLLINGER POSITION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.005  # Within 0.5% of lower band
        near_bb_upper = close[i] >= bb_upper[i] * 0.995  # Within 0.5% of upper band
        near_bb_mid = abs(close[i] - bb_mid[i]) < (bb_upper[i] - bb_mid[i]) * 0.3  # Near middle
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) — Trend Following ===
        if is_trending:
            # Long: HMA cross + RSI filter + macro bullish
            if hma_cross_long and rsi_not_overbought and macro_bull:
                desired_signal = BASE_SIZE
            # Also enter if HMA already bullish + pullback to BB mid
            elif hma_bull and macro_bull and near_bb_mid and rsi_not_overbought:
                desired_signal = REDUCED_SIZE
            
            # Short: HMA cross + RSI filter + macro bearish
            elif hma_cross_short and rsi_not_oversold and macro_bear:
                desired_signal = -BASE_SIZE
            # Also enter if HMA already bearish + retracement to BB mid
            elif hma_bear and macro_bear and near_bb_mid and rsi_not_oversold:
                desired_signal = -REDUCED_SIZE
        
        # === CHOPPY REGIME (CHOP > 61.8) — Mean Reversion ===
        elif is_choppy:
            # Long: Price at BB lower + RSI oversold + macro not strongly bearish
            if near_bb_lower and rsi[i] < 40 and not macro_bear:
                desired_signal = BASE_SIZE
            elif near_bb_lower and rsi[i] < 35:  # Strong oversold regardless of macro
                desired_signal = REDUCED_SIZE
            
            # Short: Price at BB upper + RSI overbought + macro not strongly bullish
            elif near_bb_upper and rsi[i] > 60 and not macro_bull:
                desired_signal = -BASE_SIZE
            elif near_bb_upper and rsi[i] > 65:  # Strong overbought regardless of macro
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) — Combined ===
        else:
            # Long: HMA bullish + RSI ok + macro bullish OR BB mean reversion
            if hma_bull and rsi_not_overbought and macro_bull:
                desired_signal = BASE_SIZE
            elif near_bb_lower and rsi[i] < 35:
                desired_signal = REDUCED_SIZE
            
            # Short: HMA bearish + RSI ok + macro bearish OR BB mean reversion
            elif hma_bear and rsi_not_oversold and macro_bear:
                desired_signal = -BASE_SIZE
            elif near_bb_upper and rsi[i] > 65:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA still bullish or RSI not overbought
                if hma_bull and rsi[i] < 60:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HMA still bearish or RSI not oversold
                if hma_bear and rsi[i] > 40:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HMA crosses bearish OR RSI overbought + price at BB upper
            if hma_cross_short or (rsi[i] > 65 and near_bb_upper):
                desired_signal = 0.0
            # Exit if macro reverses strongly
            if macro_bear and hma_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HMA crosses bullish OR RSI oversold + price at BB lower
            if hma_cross_long or (rsi[i] < 35 and near_bb_lower):
                desired_signal = 0.0
            # Exit if macro reverses strongly
            if macro_bull and hma_bull:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals