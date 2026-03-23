#!/usr/bin/env python3
"""
Experiment #914: 4h Primary + 12h/1d HTF — Fisher Transform + Choppiness Regime

Hypothesis: After 600+ failed strategies, combining Ehlers Fisher Transform (reversal
catcher) with Choppiness Index regime detection should work across ALL symbols.

Key insights from research:
1. Fisher Transform (period=9) catches reversals in bear rallies - entry at -1.5/+1.5 crosses
2. Choppiness Index(14) switches between mean-revert (CHOP>55) and trend-follow (CHOP<45)
3. 12h HMA(21) for medium-term trend bias (direction filter)
4. 1d HMA(21) for macro regime (bull/bear market filter)
5. ATR(14) trailing stop (2.5x) for risk management
6. RELAXED entry thresholds to guarantee 30+ trades per symbol

Why this should work on 4h:
- Fisher Transform excels at catching turning points (unlike EMA crossover)
- Choppiness filter avoids trend-following in choppy markets (2022 whipsaw killer)
- Dual HTF (12h + 1d) provides stronger trend bias than single HTF
- Relaxed Fisher thresholds (-1.2/+1.2 not -1.5/+1.5) ensure trades on all symbols
- Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Critical improvements from failed experiments:
- Fisher Transform instead of RSI (better reversal detection)
- RELAXED Fisher thresholds to guarantee trades
- Hold logic maintains position through minor pullbacks
- ALL symbols MUST have positive Sharpe (no SOL-only bias)
- Use 1d HMA as macro filter: only long if price > 1d HMA in bull market

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_regime_12h1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
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
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals when Fisher crosses -1.5 (long) or +1.5 (short).
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest)
    3. Transform: 0.5 * ln((1 + x) / (1 - x))
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, trigger
    
    for i in range(period - 1, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            trigger[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price to 0-1 range
        typical = (high[i] + low[i]) / 2.0
        x = (typical - lowest) / (highest - lowest)
        
        # Clamp to avoid log(0) or log(inf)
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Trigger line (1-period lag of fisher)
        trigger[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow).
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], np.abs(high[j] - prev_close), np.abs(low[j] - prev_close))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
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

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    fisher_4h, trigger_4h = calculate_fisher_transform(high, low, period=9)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 12h HMA for medium-term trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro regime (bull/bear market)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
        if np.isnan(fisher_4h[i]) or np.isnan(trigger_4h[i]):
            continue
        if np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MEDIUM-TERM TREND (12h HTF HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === SHORT-TERM TREND FILTER (4h SMA50/200) ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === FISHER TRANSFORM SIGNALS (Relaxed thresholds: -1.2/+1.2) ===
        fisher_cross_long = (fisher_4h[i] > -1.2) and (trigger_4h[i] <= -1.2)
        fisher_cross_short = (fisher_4h[i] < 1.2) and (trigger_4h[i] >= 1.2)
        
        # Extreme Fisher levels (stronger signals)
        fisher_extreme_long = fisher_4h[i] < -1.5
        fisher_extreme_short = fisher_4h[i] > 1.5
        
        # Fisher recovering from extreme
        fisher_recovering_long = (fisher_4h[i] > -1.0) and (fisher_4h[i-1] <= -1.0) if i > 0 else False
        fisher_weakening_short = (fisher_4h[i] < 1.0) and (fisher_4h[i-1] >= 1.0) if i > 0 else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 55) — Mean Reversion with Fisher ===
        if ranging_regime:
            # Long: Fisher cross up from oversold + trend alignment
            if fisher_cross_long and (macro_bull or trend_12h_bullish or above_sma50):
                desired_signal = BASE_SIZE
            
            # Short: Fisher cross down from overbought + trend alignment
            if fisher_cross_short and (macro_bear or trend_12h_bearish or below_sma50):
                desired_signal = -BASE_SIZE
            
            # Fallback: Extreme Fisher alone (guarantees trades)
            if fisher_extreme_long and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if fisher_extreme_short and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
            
            # Secondary fallback: Fisher recovering from extreme
            if fisher_recovering_long and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if fisher_weakening_short and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Bullish trend + Fisher recovering
            if macro_bull or trend_12h_bullish or above_sma50:
                if fisher_cross_long:
                    desired_signal = BASE_SIZE
                elif fisher_recovering_long:
                    desired_signal = REDUCED_SIZE
            
            # Short: Bearish trend + Fisher weakening
            if macro_bear or trend_12h_bearish or below_sma50:
                if fisher_cross_short:
                    desired_signal = -BASE_SIZE
                elif fisher_weakening_short:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Fisher + trend confluence
            if fisher_cross_long and (macro_bull or trend_12h_bullish):
                desired_signal = REDUCED_SIZE
            
            if fisher_cross_short and (macro_bear or trend_12h_bearish):
                desired_signal = -REDUCED_SIZE
            
            # Fallback: Extreme Fisher with SMA200 filter
            if fisher_extreme_long and above_sma200 and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if fisher_extreme_short and below_sma200 and desired_signal == 0:
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
                # Hold long if trend intact and Fisher not overbought
                if (macro_bull or trend_12h_bullish) and fisher_4h[i] < 1.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and Fisher not oversold
                if (macro_bear or trend_12h_bearish) and fisher_4h[i] > -1.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro + medium trend reverses + Fisher overbought
            if macro_bear and trend_12h_bearish and fisher_4h[i] > 1.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro + medium trend reverses + Fisher oversold
            if macro_bull and trend_12h_bullish and fisher_4h[i] < -1.5:
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