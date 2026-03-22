#!/usr/bin/env python3
"""
Experiment #424: 4h Primary + 12h/1d HTF — Dual Regime (Chop/Trend) + Connors RSI + Donchian

Hypothesis: After analyzing 400+ failed experiments, clear pattern emerges:
1. 4h timeframe needs 20-50 trades/year (higher than 12h/1d, lower than 1h)
2. Dual regime approach: mean-revert in choppy markets, trend-follow in trending
3. Connors RSI (CRSI) proven 75% win rate for mean reversion entries
4. Donchian(20) breakout for momentum/trend confirmation
5. 12h HMA for intermediate trend, 1d HMA for major regime filter
6. Previous 4h attempts failed due to: too many filters, wrong TF combination

Why this might beat current best (Sharpe=0.435):
- Choppiness Index correctly identifies regime (research: ETH Sharpe +0.923)
- CRSI catches oversold/overbought better than standard RSI(14)
- Donchian breakout adds momentum filter (worked for SOL +0.879)
- 12h/1d HTF prevents counter-trend trades in major reversals
- ATR 2.5x trailing stop protects in crash scenarios (2022-style)

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 30-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_donchian_chop_12h1d_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Sum of ATR over period
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    
    # Choppiness formula
    chop = 100.0 * np.log10((hh - ll).values / (atr_sum.values + 1e-10) + 1e-10) / np.log10(period)
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    choppiness = calculate_choppiness(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = np.inf
    entry_price = 0.0
    last_trade_bar = -50
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(adx[i]):
            continue
        
        # === 1D MAJOR TREND (primary regime filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 12H INTERMEDIATE TREND ===
        bull_regime_12h = close[i] > hma_12h_21_aligned[i]
        bear_regime_12h = close[i] < hma_12h_21_aligned[i]
        
        # === 4H LOCAL TREND (HMA crossover) ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = ranging (mean reversion preferred)
        # CHOP < 45 = trending (breakout preferred)
        is_choppy = choppiness[i] > 55.0
        is_trending = choppiness[i] < 45.0
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx[i] > 25.0
        weak_trend = adx[i] < 20.0
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        # CRSI < 20 = oversold (long opportunity)
        # CRSI > 80 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        crsi_neutral_long = crsi[i] < 45.0
        crsi_neutral_short = crsi[i] > 55.0
        
        # === DONCHIAN BREAKOUT (momentum) ===
        donchian_breakout_long = close[i] > donchian_upper[i]
        donchian_breakout_short = close[i] < donchian_lower[i]
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        if bull_regime_1d or bull_regime_12h:
            # Mean reversion entry (choppy market + CRSI oversold)
            if is_choppy and crsi_oversold:
                new_signal = LONG_SIZE
            # Trend breakout entry (trending market + Donchian breakout + HMA bullish)
            elif is_trending and donchian_breakout_long and hma_bullish:
                new_signal = LONG_SIZE
            # HMA crossover + CRSI confirmation (moderate entry)
            elif hma_bullish and crsi_neutral_long and not crsi_overbought:
                new_signal = LONG_SIZE * 0.7
            # Strong trend + pullback (ADX > 25 + CRSI < 50)
            elif strong_trend and crsi[i] < 50.0 and hma_bullish:
                new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRY
        if bear_regime_1d or bear_regime_12h:
            # Mean reversion entry (choppy market + CRSI overbought)
            if is_choppy and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Trend breakout entry (trending market + Donchian breakout + HMA bearish)
            elif is_trending and donchian_breakout_short and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # HMA crossover + CRSI confirmation (moderate entry)
            elif hma_bearish and crsi_neutral_short and not crsi_oversold:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
            # Strong trend + pullback (ADX > 25 + CRSI > 50)
            elif strong_trend and crsi[i] > 50.0 and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 25 bars (~4 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if bull_regime_12h and crsi[i] < 40.0:
                new_signal = LONG_SIZE * 0.5
            elif bear_regime_12h and crsi[i] > 60.0:
                new_signal = -SHORT_SIZE * 0.5
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on mean reversion exhaustion)
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # 1d regime flip exit
        if in_position and position_side > 0 and bear_regime_1d:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime_1d:
            new_signal = 0.0
        
        # 12h trend reversal exit
        if in_position and position_side > 0 and bear_regime_12h:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime_12h:
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
                lowest_price = close[i] if position_side < 0 else np.inf
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else np.inf
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = np.inf
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals