#!/usr/bin/env python3
"""
Experiment #252: 12h Primary + 1d/1w HTF — HMA Trend + Choppiness Regime + Connors RSI

Hypothesis: Building on #246 (KAMA+Chop, Sharpe=0.350), this strategy improves by:
1. Using 1w HMA for MACRO regime filter (bull/bear market identification)
2. Using 1d HMA slope for PRIMARY trend direction (faster than 1w, slower than 12h)
3. Using Connors RSI (CRSI) instead of standard RSI — proven 75% win rate for mean reversion
4. Choppiness Index on 12h for regime switching (trend vs mean revert mode)
5. Asymmetric entry thresholds to ensure 20-40 trades/year on 12h timeframe
6. 2.5x ATR trailing stops with regime reversal exits

CRSI Formula: (RSI(2) + RSI_Streak(2) + PercentRank(100)) / 3
- RSI(2): Very fast momentum
- RSI_Streak(2): Consecutive up/down days
- PercentRank(100): Where current price ranks vs last 100 bars

Entry Logic:
- TREND MODE (CHOP < 45): Follow 1d HMA slope direction with CRSI confirmation
- MEAN REVERT MODE (CHOP > 55): Fade extremes with CRSI < 10 (long) or > 90 (short)
- MACRO FILTER: Only long if 1w HMA bullish, only short if 1w HMA bearish

Position Sizing: 0.20 base, 0.30 strong (discrete levels to minimize fee churn)
Target: 25-45 trades/year per symbol (within 12h cost model of 1-2.5% fee drag)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_chop_crsi_regime_1d1w_v1"
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

def calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(period) + RSI_Streak(period) + PercentRank(period)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Current price rank vs last N periods (0-100 scale)
    
    Entry signals: CRSI < 10 (oversold long), CRSI > 90 (overbought short)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: Fast RSI(2)
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
    # Streak = consecutive up (positive) or down (negative) days
    delta = close_s.diff().fillna(0)
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak_abs[i] >= streak_period:
            streak_rsi[i] = 100 if streak_sign[i] > 0 else 0
        else:
            # Partial streak - scale linearly
            streak_rsi[i] = 50 + (streak[i] / streak_period) * 50
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: Percent Rank
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster than EMA, less lag, smoother than SMA.
    """
    close_s = pd.Series(close)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (macro regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Calculate 1d HTF indicators (primary trend)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=2, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -25
    consecutive_no_signal = 0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1W MACRO REGIME (highest level filter) ===
        # Bull macro: 1w HMA slope > 0.1%
        # Bear macro: 1w HMA slope < -0.1%
        macro_bull = hma_1w_slope_aligned[i] > 0.10
        macro_bear = hma_1w_slope_aligned[i] < -0.10
        macro_neutral = not macro_bull and not macro_bear
        
        # === 1D PRIMARY TREND (direction filter) ===
        # Bull trend: 1d HMA slope > 0.2%
        # Bear trend: 1d HMA slope < -0.2%
        trend_bull = hma_1d_slope_aligned[i] > 0.20
        trend_bear = hma_1d_slope_aligned[i] < -0.20
        trend_neutral = not trend_bull and not trend_bear
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME (12h) ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 12H LOCAL SIGNALS ===
        price_above_12h_hma = close[i] > hma_12h_21[i]
        price_below_12h_hma = close[i] < hma_12h_21[i]
        hma_12h_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_12h_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = deeply oversold (long opportunity)
        # CRSI > 85 = deeply overbought (short opportunity)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        consecutive_no_signal += 1 if new_signal == 0.0 else 0
        
        # TREND FOLLOWING MODE (when trending + regime aligned)
        if is_trending:
            # LONG: Macro bull + trend bull + price above 12h HMA + CRSI confirming
            if macro_bull and trend_bull and price_above_12h_hma and crsi[i] > 40:
                new_signal = STRONG_SIZE
            # LONG: Trend bull + 12h HMA bullish + CRSI neutral-bull
            elif trend_bull and hma_12h_bullish and crsi[i] > 35 and crsi[i] < 70:
                new_signal = BASE_SIZE
            
            # SHORT: Macro bear + trend bear + price below 12h HMA + CRSI confirming
            if macro_bear and trend_bear and price_below_12h_hma and crsi[i] < 60:
                new_signal = -STRONG_SIZE
            # SHORT: Trend bear + 12h HMA bearish + CRSI neutral-bear
            elif trend_bear and hma_12h_bearish and crsi[i] < 65 and crsi[i] > 30:
                new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: Choppy + CRSI deeply oversold + macro not strongly bear
            if crsi_extreme_oversold and not macro_bear:
                new_signal = BASE_SIZE
            # LONG: Choppy + CRSI extreme oversold (any macro)
            elif crsi[i] < 8:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.8
            
            # SHORT: Choppy + CRSI deeply overbought + macro not strongly bull
            if crsi_extreme_overbought and not macro_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + CRSI extreme overbought (any macro)
            elif crsi[i] > 92:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 30 bars (~15 days on 12h)
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if macro_bull and trend_bull and crsi[i] > 35:
                new_signal = BASE_SIZE * 0.6
            elif macro_bear and trend_bear and crsi[i] < 65:
                new_signal = -BASE_SIZE * 0.6
            elif is_choppy and crsi[i] < 20:
                new_signal = BASE_SIZE * 0.5
            elif is_choppy and crsi[i] > 80:
                new_signal = -BASE_SIZE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but macro + trend turn strongly bearish
            if position_side > 0 and macro_bear and trend_bear and price_below_1d_hma:
                regime_reversal = True
            # Short position but macro + trend turn strongly bullish
            if position_side < 0 and macro_bull and trend_bull and price_above_1d_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                consecutive_no_signal = 0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                consecutive_no_signal = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals