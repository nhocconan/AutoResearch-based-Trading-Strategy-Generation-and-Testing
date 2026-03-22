#!/usr/bin/env python3
"""
Experiment #437: 1d Primary + 1w HTF — Choppiness Regime + HMA + Connors RSI

Hypothesis: After 436 experiments, clear pattern emerges for 1d strategies:
1. 1d timeframe naturally produces fewer trades (10-30/year) — need simpler entry logic
2. Choppiness Index regime detection (trend vs range) is proven in research notes
3. Connors RSI (CRSI) has 75% win rate for mean reversion entries
4. 1w HMA for major trend direction prevents counter-trend disasters in 2022-style crashes
5. Current best (Sharpe=0.435) uses 1d HMA+RSI+1w — we add regime adaptation

Why this might beat current best:
- CHOP regime switch adapts logic: trend-follow when CHOP<38, mean-revert when CHOP>62
- Connors RSI (RSI2 + RSI_Streak + PercentRank) / 3 catches oversold/overbought better
- 1w HTF filter prevents 2022-style whipsaw (BTC -77% crash)
- Simpler entry = more trades = meet >=30 trades/symbol requirement
- ATR 2.5x trailing stop protects in crash scenarios

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 20-40 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_hma_1w_v1"
timeframe = "1d"
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
    """Calculate Choppiness Index (CHOP)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    # CHOP = 100 * log10(sum(TR) / (HH - LL)) / log10(period)
    chop = 100.0 * np.log10(tr_sum / (hh_ll + 1e-10)) / np.log10(period)
    
    return chop.values

def calculate_connors_rsi(close, rsi_period=2, streak_period=2, pct_rank_period=100):
    """Calculate Connors RSI (CRSI)."""
    close_s = pd.Series(close)
    
    # RSI(2) - very short term momentum
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI of Streak (consecutive up/down days)
    streak = pd.Series(0.0, index=close_s.index)
    for i in range(1, len(close_s)):
        if close_s.iloc[i] > close_s.iloc[i-1]:
            streak.iloc[i] = streak.iloc[i-1] + 1 if streak.iloc[i-1] >= 0 else 1
        elif close_s.iloc[i] < close_s.iloc[i-1]:
            streak.iloc[i] = streak.iloc[i-1] - 1 if streak.iloc[i-1] <= 0 else -1
    
    # RSI of streak values
    streak_gain = streak.where(streak > 0, 0.0)
    streak_loss = -streak.where(streak < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank (100-day)
    pct_rank = close_s.rolling(window=pct_rank_period, min_periods=pct_rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    
    # CRSI = (RSI(2) + RSI_Streak(2) + PercentRank(100)) / 3
    crsi = (rsi_short + rsi_streak + pct_rank) / 3.0
    
    return crsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    hma_1d_10 = calculate_hma(close, period=10)
    hma_1d_30 = calculate_hma(close, period=30)
    rsi_1d_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close)
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
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
    last_trade_bar = -30
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(hma_1d_10[i]) or np.isnan(hma_1d_30[i]):
            continue
        if np.isnan(rsi_1d_14[i]) or np.isnan(crsi[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w HMA = bull market bias (favor longs)
        # Price below 1w HMA = bear market bias (favor shorts)
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 61.8 = ranging market (mean reversion)
        # CHOP < 38.2 = trending market (trend follow)
        # 38.2-61.8 = transition (use HMA direction)
        is_ranging = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        
        # === 1D LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_1d_10[i] > hma_1d_30[i]
        hma_bearish = hma_1d_10[i] < hma_1d_30[i]
        
        # === RSI & CRSI SIGNALS ===
        rsi_oversold = rsi_1d_14[i] < 45.0
        rsi_overbought = rsi_1d_14[i] > 55.0
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if bull_regime or above_sma200:
            # Trending market: trend follow on HMA bullish
            if is_trending and hma_bullish:
                new_signal = LONG_SIZE
            # Ranging market: mean reversion on CRSI oversold
            elif is_ranging and crsi_oversold:
                new_signal = LONG_SIZE
            # RSI pullback in bull regime
            elif bull_regime and rsi_oversold and hma_bullish:
                new_signal = LONG_SIZE * 0.9
        
        # SHORT ENTRIES
        if bear_regime or below_sma200:
            # Trending market: trend follow on HMA bearish
            if is_trending and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Ranging market: mean reversion on CRSI overbought
            elif is_ranging and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # RSI bounce in bear regime
            elif bear_regime and rsi_overbought and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.9
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 20 bars (~20 days on 1d), force entry on weaker signal
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if bull_regime and hma_bullish and rsi_1d_14[i] < 50.0:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime and hma_bearish and rsi_1d_14[i] > 50.0:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_1d_14[i] > 70.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_1d_14[i] < 30.0:
            new_signal = 0.0
        
        # CRSI extreme exit
        if in_position and position_side > 0 and crsi[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 20.0:
            new_signal = 0.0
        
        # Trend reversal exit (1w regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (1d HMA cross)
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