#!/usr/bin/env python3
"""
Experiment #979: 4h Primary + 1d HTF — Ehlers Fisher Transform + Regime Adaptive + HMA Trend

Hypothesis: After 705+ failed strategies, the Ehlers Fisher Transform excels at catching
reversals in bear/range markets (2022 crash, 2025 bear) where RSI/EMA strategies fail.

Why this should work:
1. Fisher Transform converts price to Gaussian distribution → cleaner extreme signals
2. Long when Fisher crosses above -1.5 (oversold reversal), short when crosses below +1.5
3. 1d HMA(21) for macro trend bias (only trade with HTF trend)
4. Choppiness Index regime switch: CHOP>55 = mean revert, CHOP<45 = trend follow
5. ATR(14) trailing stop at 2.5x for risk management

Key improvements over failed strategies:
- NO funding rates (data alignment caused 0 trades in #970, #974)
- NO CRSI (failed on BTC in multiple experiments)
- RELAXED Fisher thresholds (-1.5/+1.5 not -2.0/+2.0) to ensure trades
- Only 2 confluence factors required (not 4-5 which caused 0 trades)
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
Position size: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_regime_1d_hma_chop_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — converts price to Gaussian distribution.
    Reference: Ehlers, J.F. (2002) "Fishing With A Transform"
    
    Long signal: Fisher crosses above -1.5 (oversold reversal)
    Short signal: Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    # Calculate price bound (median of high-low range)
    for i in range(period, n):
        # Use typical price
        typical = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * (typical - lowest_low) / range_val - 1.0
        
        # Clamp to avoid extreme values
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform: 0.5 * ln((1+x)/(1-x))
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        # Signal line (1-period lag of fisher)
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_hma(series, period):
    """Hull Moving Average — responsive trend indicator."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    Reference: E.W. Dreiss
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending
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
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, close, period=9)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    
    # Calculate and align 1d HMA for macro regime (Rule 2 - use align_htf_to_ltf)
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
    
    # Track Fisher crossings
    prev_fisher = np.nan
    prev_fisher_signal = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            prev_fisher = fisher_4h[i] if not np.isnan(fisher_4h[i]) else prev_fisher
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_4h[i]):
            prev_fisher = fisher_4h[i] if not np.isnan(fisher_4h[i]) else prev_fisher
            continue
        
        # === MACRO REGIME (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 55
        trending_regime = chop_4h[i] < 45
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if not np.isnan(prev_fisher) and not np.isnan(fisher_signal_4h[i]):
            # Long: Fisher crosses above -1.5 from below
            if prev_fisher <= -1.5 and fisher_4h[i] > -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 from above
            if prev_fisher >= 1.5 and fisher_4h[i] < 1.5:
                fisher_cross_short = True
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_4h[i] < 40
        rsi_overbought = rsi_4h[i] > 60
        rsi_neutral = 40 <= rsi_4h[i] <= 60
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 55) — Mean Reversion ===
        if ranging_regime:
            # Long: Fisher oversold cross + RSI confirmation
            if fisher_cross_long and rsi_oversold:
                desired_signal = BASE_SIZE
            # Long: Fisher deeply oversold (no cross needed)
            elif fisher_4h[i] < -2.0 and rsi_4h[i] < 35:
                desired_signal = REDUCED_SIZE
            # Long: Macro bull + Fisher recovering from oversold
            elif macro_bull and fisher_4h[i] > -1.0 and fisher_4h[i] < 0:
                desired_signal = REDUCED_SIZE
            
            # Short: Fisher overbought cross + RSI confirmation
            if fisher_cross_short and rsi_overbought:
                desired_signal = -BASE_SIZE
            # Short: Fisher deeply overbought
            elif fisher_4h[i] > 2.0 and rsi_4h[i] > 65:
                desired_signal = -BASE_SIZE
            # Short: Macro bear + Fisher weakening from overbought
            elif macro_bear and fisher_4h[i] < 1.0 and fisher_4h[i] > 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — Trend Following ===
        elif trending_regime:
            # Long: Macro bull + Fisher pullback to neutral
            if macro_bull:
                if fisher_cross_long:
                    desired_signal = BASE_SIZE
                elif fisher_4h[i] < -0.5 and rsi_oversold:
                    desired_signal = REDUCED_SIZE
            
            # Short: Macro bear + Fisher rally to neutral
            if macro_bear:
                if fisher_cross_short:
                    desired_signal = -BASE_SIZE
                elif fisher_4h[i] > 0.5 and rsi_overbought:
                    desired_signal = -REDUCED_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: Only take Fisher extremes with macro confluence
            if fisher_cross_long and macro_bull:
                desired_signal = BASE_SIZE
            elif fisher_cross_long and fisher_4h[i] < -1.8:
                desired_signal = REDUCED_SIZE
            
            if fisher_cross_short and macro_bear:
                desired_signal = -BASE_SIZE
            elif fisher_cross_short and fisher_4h[i] > 1.8:
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
                # Hold long if macro bull and Fisher not overbought
                if macro_bull and fisher_4h[i] < 1.5:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro bear and Fisher not oversold
                if macro_bear and fisher_4h[i] > -1.5:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses to bear + Fisher overbought
            if macro_bear and fisher_4h[i] > 1.5:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_4h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses to bull + Fisher oversold
            if macro_bull and fisher_4h[i] < -1.5:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_4h[i] < 25:
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
        
        # Update previous Fisher for next iteration
        prev_fisher = fisher_4h[i]
    
    return signals