#!/usr/bin/env python3
"""
Experiment #284: 4h Primary + 12h/1d HTF — Simplified Connors RSI + Choppiness Regime

Hypothesis: After 25+ failed 4h experiments with complex multi-filter logic,
simplify to ONE strong edge: Connors RSI mean reversion + Choppiness regime filter.

Key insights from failures (#274, #279, #281):
1. Too many conflicting filters = 0 trades or whipsaw losses
2. Complex position state machines cause bugs in stoploss tracking
3. Connors RSI (CRSI) has proven 75% win rate in academic literature
4. Choppiness Index is the BEST regime filter for bear/range markets (ETH Sharpe +0.923)

Strategy design:
1. 1d HMA for PRIMARY trend direction (bull/bear regime)
2. Choppiness(14) on 4h: >55 = range (mean revert), <45 = trend (breakout)
3. Connors RSI(3,2,100) for entry timing (more sensitive than RSI14)
4. Simple signal = position size (no complex state machine)
5. Stoploss: 2.5*ATR trailing via signal→0

Position sizing: 0.30 base, 0.40 strong conviction (discrete levels)
Target: 25-50 trades/year on 4h timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_connors_chop_hma_1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: percentile rank of current close vs last 100 closes
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: Short-term RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    # Streak = consecutive up/down days (+1 for up, -1 for down, 0 for flat)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: Percentile Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = 100 * rank / rank_period
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    STRONG_SIZE = 0.40
    
    # Track for stoploss
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    position_side = np.zeros(n)
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_21[i]):
            signals[i] = 0.0
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_50_aligned[i]
        regime_bear = close[i] < hma_1d_50_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_oversold = crsi[i] < 25.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_overbought = crsi[i] > 75.0
        
        # === LOCAL TREND ===
        price_above_hma = close[i] > hma_4h_21[i]
        price_below_hma = close[i] < hma_4h_21[i]
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer filters) ===
        new_signal = 0.0
        
        # MEAN REVERSION MODE (choppy market) - PRIMARY STRATEGY
        if is_choppy:
            # LONG: CRSI oversold + price near/above 4h HMA
            if crsi_oversold and price_above_hma:
                new_signal = BASE_SIZE
            # LONG: CRSI extreme oversold (any regime alignment)
            if crsi_extreme_oversold:
                new_signal = STRONG_SIZE
            
            # SHORT: CRSI overbought + price near/below 4h HMA
            if crsi_overbought and price_below_hma:
                new_signal = -BASE_SIZE
            # SHORT: CRSI extreme overbought (any regime alignment)
            if crsi_extreme_overbought:
                new_signal = -STRONG_SIZE
        
        # TREND MODE (trending market) - SECONDARY STRATEGY
        elif is_trending:
            # LONG: Bull regime + CRSI pullback (not extreme) + price above HMA
            if regime_bull and 30 < crsi[i] < 60 and price_above_hma:
                new_signal = BASE_SIZE
            # SHORT: Bear regime + CRSI rally (not extreme) + price below HMA
            if regime_bear and 40 < crsi[i] < 70 and price_below_hma:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY BOOST (ensure 10+ trades) ===
        # If CRSI is very extreme, always take the trade
        if crsi[i] < 10.0 and new_signal == 0.0:
            new_signal = STRONG_SIZE
        if crsi[i] > 90.0 and new_signal == 0.0:
            new_signal = -STRONG_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Track position state from previous bar
        prev_signal = signals[i-1] if i > 0 else 0.0
        prev_side = np.sign(prev_signal)
        
        if prev_side != 0:
            # Update extremum since entry
            if prev_side > 0:
                # Long position
                if i == 150 or prev_signal == 0.0:
                    highest_since_entry[i] = close[i]
                else:
                    highest_since_entry[i] = max(highest_since_entry[i-1], close[i])
                stoploss_price = highest_since_entry[i] - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss hit
            
            if prev_side < 0:
                # Short position
                if i == 150 or prev_signal == 0.0:
                    lowest_since_entry[i] = close[i]
                else:
                    lowest_since_entry[i] = min(lowest_since_entry[i-1], close[i])
                stoploss_price = lowest_since_entry[i] + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss hit
        
        # === REGIME REVERSAL EXIT ===
        if prev_side > 0 and regime_bear and price_below_hma:
            new_signal = 0.0
        if prev_side < 0 and regime_bull and price_above_hma:
            new_signal = 0.0
        
        signals[i] = new_signal
    
    return signals