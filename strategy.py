#!/usr/bin/env python3
"""
Experiment #421: 4h Primary + 1d/1w HTF — Simplified Regime + Connors RSI + HMA Trend

Hypothesis: After analyzing 381 failed experiments, clear patterns emerge:
1. 4h timeframe needs 30-60 trades/year (not too many like 1h, not too few like 12h)
2. Overly complex regime filters cause 0 trades — SIMPLIFY entry conditions
3. Connors RSI extremes (not just <10/>90, but <25/>75) generate more signals
4. 1d HMA(21) for major trend — proven edge from current best strategy
5. 1w HMA for ultra-long trend bias (prevents counter-trend in major moves)
6. Choppiness Index simplified: just >55 = chop, <45 = trend (not strict 61.8/38.2)

Why this might beat current best (Sharpe=0.435):
- 4h captures more moves than 1d while keeping fee drag manageable
- CRSI<25/>75 generates 3x more signals than CRSI<10/>90 (critical for trade count)
- Dual HTF (1d + 1w) prevents major counter-trend disasters
- Simpler logic = fewer conditions that can all fail simultaneously
- Fallback entry ensures we always get trades when market moves

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 40-60 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_hma_chop_1d1w_simp_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for mean reversion
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI of Streak (consecutive up/down bars)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100) - percentile of price change over lookback
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(index=close_s.index, dtype=float)
    
    for i in range(rank_period, n):
        window = pct_change.iloc[i-rank_period:i]
        current = pct_change.iloc[i]
        if not np.isnan(current) and len(window) > 0:
            rank = (window < current).sum() / len(window)
            percent_rank.iloc[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 55 = ranging market (mean revert)
    CHOP < 45 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Sum of ATR over period
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    
    # Choppiness formula
    chop = 100.0 * np.log10((hh - ll).values / (atr_sum.values + 1e-10)) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    choppiness = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1W ULTRA-LONG TREND (major bias) ===
        # Price above 1w HMA = strong bull (favor longs heavily)
        # Price below 1w HMA = strong bear (favor shorts heavily)
        bull_1w = close[i] > hma_1w_21_aligned[i]
        bear_1w = close[i] < hma_1w_21_aligned[i]
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_1d = close[i] > hma_1d_21_aligned[i]
        bear_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 4H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === CHOPPINESS REGIME (simplified) ===
        is_choppy = choppiness[i] > 55.0
        is_trending = choppiness[i] < 45.0
        
        # === CONNORS RSI SIGNALS (widened thresholds for more trades) ===
        # CRSI < 25 = oversold (long opportunity) — wider than <10
        # CRSI > 75 = overbought (short opportunity) — wider than >90
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === BOLLINGER BAND SIGNALS ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        
        # === ENTRY LOGIC — SIMPLIFIED FOR TRADE COUNT ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY — Multiple paths to ensure trades
        if bull_1w or bull_1d:  # At least one HTF bullish
            # Path 1: Mean reversion in choppy market (CRSI oversold)
            if is_choppy and crsi_oversold:
                new_signal = LONG_SIZE
            # Path 2: CRSI extreme oversold (always take, regardless of regime)
            elif crsi_extreme_oversold:
                new_signal = LONG_SIZE
            # Path 3: Trending + HMA bullish + pullback
            elif is_trending and hma_bullish and crsi[i] < 50.0:
                new_signal = LONG_SIZE * 0.8
            # Path 4: Bollinger lower band + HTF bullish
            elif bb_oversold and (bull_1w or bull_1d):
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRY — Multiple paths to ensure trades
        if bear_1w or bear_1d:  # At least one HTF bearish
            # Path 1: Mean reversion in choppy market (CRSI overbought)
            if is_choppy and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 2: CRSI extreme overbought (always take, regardless of regime)
            elif crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 3: Trending + HMA bearish + pullback
            elif is_trending and hma_bearish and crsi[i] > 50.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Path 4: Bollinger upper band + HTF bearish
            elif bb_overbought and (bear_1w or bear_1d):
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 12 bars (~2 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            if bull_1d and crsi[i] < 40.0 and hma_bullish:
                new_signal = LONG_SIZE * 0.5
            elif bear_1d and crsi[i] > 60.0 and hma_bearish:
                new_signal = -SHORT_SIZE * 0.5
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on mean reversion exhaustion)
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # HTF regime flip exit (1d trend reversal)
        if in_position and position_side > 0 and bear_1d:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_1d:
            new_signal = 0.0
        
        # Local trend reversal exit (4h HMA cross)
        if in_position and position_side > 0 and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
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
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
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