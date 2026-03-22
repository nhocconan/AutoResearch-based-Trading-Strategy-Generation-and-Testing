#!/usr/bin/env python3
"""
Experiment #557: 12h Fisher Transform Reversal with Dual HTF Regime Filter

Hypothesis: After analyzing 500+ failed experiments, the key insight is:
1. 12h timeframe is underutilized - less noise than 15m/30m/1h/4h (all failed)
2. Fisher Transform excels at catching reversals in bear/range markets (2022, 2025)
3. Dual HTF filter (1d HMA + 1w HMA) provides stronger regime confirmation
4. ADX hysteresis (enter >25, exit <18) prevents chop whipsaw that destroyed exp 545/554
5. Asymmetric sizing: 0.30 with trend, 0.15 against trend (reduces counter-trend risk)
6. 3*ATR stoploss for 12h (wider than 2.5*ATR) avoids premature exits on multi-day swings

Why Fisher Transform over RSI:
- Fisher normalizes price to Gaussian distribution, better for extreme detection
- RSI failed repeatedly (exp 547, 548, 553 all negative Sharpe)
- Fisher crosses at -1.5/+1.5 are cleaner signals than RSI 30/70
- Works well in both trending AND ranging markets (critical for 2025 bear)

Why 12h specifically:
- 2 bars/day = ~730 bars/year = manageable frequency
- Captures multi-day swings without intraday noise
- Less fee drag than 15m/30m/1h strategies
- Better suited for crypto's multi-day trend cycles

Timeframe: 12h (REQUIRED)
HTF: 1d + 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.15-0.30 discrete (max 0.40)
Stoploss: 3.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_reversal_dual_htf_regime_adx_hysteresis_atr_v1"
timeframe = "12h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Normalize price to range -1 to +1
    highest = hl2_s.rolling(window=period, min_periods=period).max()
    lowest = hl2_s.rolling(window=period, min_periods=period).min()
    range_val = highest - lowest
    range_val = range_val.replace(0, np.inf)  # avoid division by zero
    
    normalized = 0.66 * ((hl2 - lowest) / range_val - 0.5) + 0.67 * np.roll(normalized if 'normalized' in dir() else 0, 1)
    
    # Simpler approach: use close-based Fisher
    close_s = pd.Series(close)
    highest_close = close_s.rolling(window=period, min_periods=period).max()
    lowest_close = close_s.rolling(window=period, min_periods=period).min()
    range_close = highest_close - lowest_close
    range_close = range_close.replace(0, 0.0001)
    
    normalized = 0.66 * ((close - lowest_close) / range_close - 0.5) + 0.67 * 0
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 0.0001))
    fisher = pd.Series(fisher).ewm(span=3, min_periods=3, adjust=False).mean()
    
    return fisher.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    fisher = calculate_fisher_transform(high, low, close, 9)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels with asymmetry (Rule 4)
    SIZE_WITH_TREND = 0.30
    SIZE_COUNTER_TREND = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # ADX hysteresis state
    adx_above_entry = False  # ADX crossed above 25
    adx_below_exit = False   # ADX crossed below 18
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL HTF REGIME FILTER ===
        # 1d HMA: intermediate trend
        bull_1d = close[i] > hma_1d_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        
        # 1w HMA: long-term regime (bull/bear market)
        bull_1w = close[i] > hma_1w_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bull: both 1d and 1w bullish
        strong_bull = bull_1d and bull_1w
        # Strong bear: both 1d and 1w bearish
        strong_bear = bear_1d and bear_1w
        # Mixed regime: conflicting signals (reduce size or stay flat)
        mixed_regime = (bull_1d and bear_1w) or (bear_1d and bull_1w)
        
        # === ADX HYSTERESIS (avoid chop whipsaw) ===
        if adx_14[i] > 25:
            adx_above_entry = True
        if adx_14[i] < 18:
            adx_above_entry = False
            adx_below_exit = True
        if adx_14[i] > 20:
            adx_below_exit = False
        
        trend_allowed = adx_above_entry or (in_position and not adx_below_exit)
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = fisher[i] < -1.5 and (i > 0 and fisher[i-1] >= fisher[i])
        fisher_short = fisher[i] > 1.5 and (i > 0 and fisher[i-1] <= fisher[i])
        
        # Check for Fisher cross (more precise entry)
        fisher_cross_long = False
        fisher_cross_short = False
        if i > 0 and not np.isnan(fisher[i-1]):
            # Long: Fisher crosses above -1.5 from below
            if fisher[i-1] < -1.5 and fisher[i] >= -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 from above
            if fisher[i-1] > 1.5 and fisher[i] <= 1.5:
                fisher_cross_short = True
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long entry: Fisher cross + regime filter + ADX allows
        if fisher_cross_long and trend_allowed:
            if strong_bull:
                new_signal = SIZE_WITH_TREND
            elif not mixed_regime:
                new_signal = SIZE_COUNTER_TREND
            # mixed_regime = no entry
        
        # Short entry: Fisher cross + regime filter + ADX allows
        elif fisher_cross_short and trend_allowed:
            if strong_bear:
                new_signal = -SIZE_WITH_TREND
            elif not mixed_regime:
                new_signal = -SIZE_COUNTER_TREND
            # mixed_regime = no entry
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME REVERSAL EXIT ===
        # Exit if 1w HMA flips against position (major regime change)
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_1w:
                new_signal = 0.0
            if position_side < 0 and bull_1w:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals