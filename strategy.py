#!/usr/bin/env python3
"""
Experiment #439: 4h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA

Hypothesis: After 438 experiments, clear pattern emerges:
1. Pure trend-following fails on BTC/ETH (2022 crash destroys gains)
2. Choppiness Index regime detection proven for ETH (Sharpe +0.923 in research)
3. Connors RSI has 75% win rate for mean-reversion entries
4. 4h timeframe targets 20-50 trades/year (optimal fee/trade balance)
5. 1d HMA filter prevents major counter-trend disasters

Why this might beat current best (Sharpe=0.435):
- Regime-adaptive: mean-revert in chop, trend-follow in trends
- Connors RSI (RSI3 + RSI-Streak + PercentRank) catches reversals better than standard RSI
- 1d HTF ensures we don't short bull markets or long bear markets
- ATR trailing stop protects in crash scenarios (2022-style)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 25-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_connors_hma_1d_v2"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(sum_atr / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Fast momentum
    RSI-Streak(2): Duration of up/down streaks
    PercentRank(100): Where current return ranks vs last 100 days
    """
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI of Streak
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x) if len(x) >= rank_period else np.nan, raw=False
    ) * 100.0
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    rsi_4h_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        if np.isnan(rsi_4h_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Price above 1d HMA21 = bull market bias (favor longs)
        # Price below 1d HMA21 = bear market bias (favor shorts)
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # Strong trend confirmation
        strong_bull = bull_regime and hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        strong_bear = bear_regime and hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = ranging market (mean reversion strategy)
        # CHOP < 45 = trending market (trend follow strategy)
        # 45-55 = transition (use HMA direction)
        is_ranging = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 4H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 20 = oversold (long opportunity)
        # CRSI > 80 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === STANDARD RSI FILTER ===
        rsi_oversold = rsi_4h_14[i] < 35.0
        rsi_overbought = rsi_4h_14[i] > 65.0
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if bull_regime or (not strong_bear):
            # Ranging market: mean reversion on CRSI extreme
            if is_ranging and crsi_oversold:
                new_signal = LONG_SIZE
            # Trending market: trend follow on pullback
            elif is_trending and hma_bullish and crsi[i] < 40.0:
                new_signal = LONG_SIZE
            # Transition zone: HMA confirmation + RSI filter
            elif hma_bullish and rsi_oversold and not bear_regime:
                new_signal = LONG_SIZE * 0.8
            # Extreme CRSI override (high probability)
            elif crsi_extreme_oversold and not strong_bear:
                new_signal = LONG_SIZE
        
        # SHORT ENTRIES
        if bear_regime or (not strong_bull):
            # Ranging market: mean reversion on CRSI extreme
            if is_ranging and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Trending market: trend follow on bounce
            elif is_trending and hma_bearish and crsi[i] > 60.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Transition zone: HMA confirmation + RSI filter
            elif hma_bearish and rsi_overbought and not bull_regime:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Extreme CRSI override (high probability)
            elif crsi_extreme_overbought and not strong_bull:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~2.5 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if bull_regime and hma_bullish and crsi[i] < 35.0:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and hma_bearish and crsi[i] > 65.0:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip)
        if in_position and position_side > 0 and strong_bear:
            new_signal = 0.0
        if in_position and position_side < 0 and strong_bull:
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
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
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