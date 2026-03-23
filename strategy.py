#!/usr/bin/env python3
"""
Experiment #661: 4h Primary + 1d HTF — Fisher Transform + KAMA Adaptive Trend + Choppiness Regime

Hypothesis: After analyzing 579 failed strategies, the key insight is:
1. Fisher Transform excels at catching reversals in bear markets (research note #3)
2. KAMA adapts smoothing based on market efficiency ratio — less whipsaw than EMA/HMA
3. #651 proved Choppiness + CRSI works on 4h (Sharpe=0.222) — keep regime filter
4. Current best (Sharpe=0.520) is 1d timeframe — adapt logic to 4h with 1d HTF bias

This strategy combines:
- Ehlers Fisher Transform (period=9): Long when crosses above -1.5, short when crosses below +1.5
- KAMA (ER=10, fast=2, slow=30): Adaptive trend following with less lag in trends, more smoothing in chop
- Choppiness Index (14): Regime filter — mean-revert in chop (>55), trend-follow when trending (<45)
- 1d HMA slope: Major trend bias to avoid counter-trend trades

Why this might beat Sharpe=0.520:
- Fisher Transform specifically designed for reversal detection in non-gaussian distributions
- KAMA's adaptive nature reduces whipsaw during 2022 crash (where EMA/HMA failed)
- Choppiness regime prevents trend-following entries in ranges (major loss source per #645-650)
- 1d HTF keeps us on right side of major moves without over-filtering
- Conservative sizing (0.28) + ATR stop controls drawdown

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 4h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_chop_1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    
    KAMA adapts smoothing based on market efficiency:
    - High efficiency (trending): uses fast SC, follows price closely
    - Low efficiency (choppy): uses slow SC, smooths out noise
    
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast - slow) + slow)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER)
    price_change = np.abs(close_s.diff(er_period))
    volatility = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (volatility + 1e-10)
    er = er.fillna(0).values
    
    # Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    
    Fisher Transform converts price to a Gaussian distribution,
    making extreme values easier to identify for reversals.
    
    Formula:
    1. Normalize price: (2 * (close - LL) / (HH - LL) - 1)
    2. Apply Fisher: 0.5 * ln((1 + value) / (1 - value))
    3. Signal line: EMA of Fisher
    
    Entry signals:
    - Long: Fisher crosses above -1.5 (extreme oversold reversal)
    - Short: Fisher crosses below +1.5 (extreme overbought reversal)
    """
    close = (high + low) / 2.0
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    hh = close_s.rolling(window=period, min_periods=period).max().values
    ll = close_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price to range [-1, +1]
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = 2.0 * (close - ll) / (hh - ll + 1e-10) - 1.0
    
    # Clip to avoid log(0) or log(negative)
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
    
    # Signal line (EMA of Fisher)
    fisher_s = pd.Series(fisher)
    fisher_signal = fisher_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Range/consolidation (mean-revert)
    - CHOP < 38.2: Trending (trend-follow)
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
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(kama_4h[i]) or np.isnan(fisher[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (HMA slope over 5 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 55.0  # Range/consolidation
        is_trend = chop_14[i] < 45.0  # Trending
        
        # === KAMA TREND ===
        kama_slope_bull = kama_4h[i] > kama_4h[i-3] if i >= 3 else False
        kama_slope_bear = kama_4h[i] < kama_4h[i-3] if i >= 3 else False
        
        price_above_kama = close[i] > kama_4h[i]
        price_below_kama = close[i] < kama_4h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher[i-1] <= fisher_signal[i-1]
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher[i-1] >= fisher_signal[i-1]
        
        fisher_oversold = fisher[i] < -1.5  # Extreme oversold reversal zone
        fisher_overbought = fisher[i] > 1.5  # Extreme overbought reversal zone
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market + Fisher oversold reversal = mean revert long
        if is_range and fisher_oversold and fisher_cross_up:
            new_signal = POSITION_SIZE
        
        # Regime 2: Trending market + 1d bull + KAMA bull + Fisher pullback entry
        elif is_trend and hma_1d_slope_bull and price_above_hma_1d:
            if kama_slope_bull and price_above_kama:
                # Enter on Fisher cross up from oversold, or pullback to KAMA
                if (fisher[i] < -0.5 and fisher_cross_up) or (close[i] < kama_4h[i] * 1.005 and close[i] > kama_4h[i] * 0.995):
                    new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market + Fisher overbought reversal = mean revert short
        elif is_range and fisher_overbought and fisher_cross_down:
            new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market + 1d bear + KAMA bear + Fisher pullback entry
        elif is_trend and hma_1d_slope_bear and price_below_hma_1d:
            if kama_slope_bear and price_below_kama:
                # Enter on Fisher cross down from overbought, or pullback to KAMA
                if (fisher[i] > 0.5 and fisher_cross_down) or (close[i] > kama_4h[i] * 0.995 and close[i] < kama_4h[i] * 1.005):
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            # Exit long if 1d trend flips bear OR KAMA flips bear
            if (hma_1d_slope_bear and price_below_hma_1d) or (kama_slope_bear and price_below_kama):
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend flips bull OR KAMA flips bull
            if (hma_1d_slope_bull and price_above_hma_1d) or (kama_slope_bull and price_above_kama):
                new_signal = 0.0
        
        # === EXIT ON FISHER EXTREME REVERSAL (take profit) ===
        if in_position and position_side > 0:
            # Long: exit if Fisher goes extreme overbought and crosses down
            if fisher[i] > 2.0 and fisher_cross_down:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Short: exit if Fisher goes extreme oversold and crosses up
            if fisher[i] < -2.0 and fisher_cross_up:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_price = 0.0
        
        signals[i] = new_signal
    
    return signals