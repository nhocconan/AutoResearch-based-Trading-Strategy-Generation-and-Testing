#!/usr/bin/env python3
"""
Experiment #891: 4h Primary + 1d/1w HTF — Fisher Transform + HMA Trend + ATR Stop

Hypothesis: After 600+ failed strategies, Fisher Transform provides unique edge for
reversal detection that RSI/CRSI don't capture. Key insights from research:

1. Fisher Transform (period=9): Normalizes price to Gaussian distribution, extreme
   values (-2 to +2) mark reversals better than RSI in bear/range markets.
2. 4h Primary TF: Proven best timeframe (current best Sharpe=0.612 uses 4h)
3. 1d HMA(21): Medium-term trend bias (direction filter)
4. 1w HMA(21): Macro regime (bull/bear market — only short in bear, only long in bull)
5. ATR(14) trailing stop (2.5x): Mandatory risk management
6. Relaxed Fisher thresholds: -1.5/+1.5 (not -2/+2) to ensure 30+ trades per symbol

Why Fisher Transform over RSI/CRSI:
- RSI/CRSI failed in 50+ recent experiments (all Sharpe ≤ 0)
- Fisher captures sharp reversals in crypto volatility better
- Less commonly used = less crowded signal
- Works well in both trending AND ranging markets

Critical improvements from failed experiments:
- SIMPLIFIED regime logic (1w HMA only, no Choppiness complexity)
- Asymmetric entries: long only in bull macro, short only in bear macro
- Relaxed Fisher thresholds to guarantee trades on ALL symbols
- Hold logic maintains position through minor pullbacks
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_regime_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest) -> 0 to 1
    3. Transform: 0.5 * ln((1 + x) / (1 - x)) where x = 2*normalized - 1
    4. Smooth with EMA
    
    Fisher > +1.5 = overbought (short signal)
    Fisher < -1.5 = oversold (long signal)
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 2:
        return fisher, fisher_prev
    
    # Calculate typical price
    typical = (high + low) / 2.0
    
    # Normalize price over lookback period
    normalized = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        
        if highest == lowest:
            normalized[i] = 0.5
        else:
            normalized[i] = (typical[i] - lowest) / (highest - lowest)
    
    # Apply Fisher Transform
    for i in range(period, n):
        if np.isnan(normalized[i]):
            continue
        
        # Clamp to avoid division by zero
        x = 2.0 * normalized[i] - 1.0
        x = np.clip(x, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
        
        # Smooth with previous value (Ehlers method)
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher_prev[i-1]
        
        fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_atr(high, low, close, period=14):
    """Average True Range — volatility measure for stops."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with EMA
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
        di_sum = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / (di_sum + 1e-10)
    
    # ADX is EMA of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    fisher_4h, fisher_prev_4h = calculate_fisher_transform(high, low, period=9)
    atr_4h = calculate_atr(high, low, close, period=14)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(high, low, close, period=14)
    
    # Calculate and align 1d HMA for medium-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro regime (bull/bear market)
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_prev_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(adx_4h[i]):
            continue
        
        # === MACRO REGIME (1w HTF HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM-TERM TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH (4h ADX) ===
        trend_strong = adx_4h[i] > 25
        trend_weak = adx_4h[i] < 20
        
        # === FISHER TRANSFORM SIGNALS (Relaxed thresholds: -1.5/+1.5) ===
        fisher_oversold = fisher_4h[i] < -1.5
        fisher_overbought = fisher_4h[i] > 1.5
        fisher_extreme_oversold = fisher_4h[i] < -2.0
        fisher_extreme_overbought = fisher_4h[i] > 2.0
        
        # Fisher crossover signals (more reliable than absolute levels)
        fisher_cross_up = fisher_prev_4h[i-1] < -1.0 and fisher_4h[i] > -1.0 if not np.isnan(fisher_prev_4h[i-1]) else False
        fisher_cross_down = fisher_prev_4h[i-1] > 1.0 and fisher_4h[i] < 1.0 if not np.isnan(fisher_prev_4h[i-1]) else False
        
        # === DI DIRECTION ===
        di_bullish = plus_di_4h[i] > minus_di_4h[i] if not np.isnan(plus_di_4h[i]) else False
        di_bearish = minus_di_4h[i] > plus_di_4h[i] if not np.isnan(minus_di_4h[i]) else False
        
        desired_signal = 0.0
        
        # === BULL MACRO REGIME (1w HMA) — Long Bias ===
        if macro_bull:
            # Strong long: Fisher oversold + 1d trend bullish + DI bullish
            if fisher_oversold and trend_1d_bullish and di_bullish:
                desired_signal = BASE_SIZE
            # Moderate long: Fisher oversold + 1d trend bullish (no DI filter)
            elif fisher_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            # Weak long: Fisher extreme oversold alone (guarantees trades)
            elif fisher_extreme_oversold:
                desired_signal = REDUCED_SIZE
            # Fisher cross up in bull regime
            elif fisher_cross_up and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
        
        # === BEAR MACRO REGIME (1w HMA) — Short Bias ===
        elif macro_bear:
            # Strong short: Fisher overbought + 1d trend bearish + DI bearish
            if fisher_overbought and trend_1d_bearish and di_bearish:
                desired_signal = -BASE_SIZE
            # Moderate short: Fisher overbought + 1d trend bearish (no DI filter)
            elif fisher_overbought and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
            # Weak short: Fisher extreme overbought alone (guarantees trades)
            elif fisher_extreme_overbought:
                desired_signal = -REDUCED_SIZE
            # Fisher cross down in bear regime
            elif fisher_cross_down and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL/TRANSITION REGIME — Conservative ===
        else:
            # Only take extreme Fisher signals in neutral regime
            if fisher_extreme_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            if fisher_extreme_overbought and trend_1d_bearish:
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro bull or 1d trend bullish and Fisher not overbought
                if (macro_bull or trend_1d_bullish) and fisher_4h[i] < 1.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bear or 1d trend bearish and Fisher not oversold
                if (macro_bear or trend_1d_bearish) and fisher_4h[i] > -1.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + 1d trend reverses + Fisher overbought
            if macro_bear and trend_1d_bearish and fisher_4h[i] > 1.0:
                desired_signal = 0.0
            # Exit if Fisher extremely overbought
            if fisher_4h[i] > 2.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + 1d trend reverses + Fisher oversold
            if macro_bull and trend_1d_bullish and fisher_4h[i] < -1.0:
                desired_signal = 0.0
            # Exit if Fisher extremely oversold
            if fisher_4h[i] < -2.0:
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