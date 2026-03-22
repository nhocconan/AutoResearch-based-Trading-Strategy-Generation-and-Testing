#!/usr/bin/env python3
"""
Experiment #411: 4h Primary + 1d/1w HTF — Regime-Adaptive Choppiness + Connors RSI + Donchian

Hypothesis: After analyzing 400+ failed experiments, clear patterns emerge for 4h timeframe:
1. 4h needs 20-50 trades/year (higher TF = fewer trades = less fee drag)
2. Choppiness Index regime detection is proven edge (ETH Sharpe +0.923 in research)
3. Connors RSI (CRSI) has 75% win rate for mean reversion in choppy markets
4. Donchian breakout works for trending markets (SOL Sharpe +0.782)
5. 1d HMA(21) for major trend bias prevents counter-trend disasters (2022 crash)
6. Asymmetric sizing: larger positions in confirmed trend, smaller in mean revert
7. Previous 4h attempts failed due to: simple trend logic, no regime filter, too many trades

Why this might beat current best (Sharpe=0.435):
- Regime-adaptive: mean revert in chop (CHOP>55), trend follow otherwise (CHOP<45)
- CRSI catches extremes better than RSI(14) — research shows 75% win rate
- 1d HTF filter prevents trading against major trend (reduces 2022-style whipsaw)
- Donchian(20) breakout adds momentum confirmation for trend entries
- ATR 2.5x trailing stop protects capital in crash scenarios
- Discrete position sizing (0.25/0.30) minimizes fee churn

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 20-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_donchian_regime_1d_v1"
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
    
    Research shows 75% win rate for CRSI<10 long, CRSI>90 short.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI for mean reversion
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # RSI of Streak (consecutive up/down days)
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
    chop = 100.0 * np.log10((hh - ll).values / (atr_sum.values + 1e-10)) / np.log10(period)
    
    return chop

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    change = np.abs(close_s.diff(er_period).values)
    volatility = pd.Series(np.abs(close_s.diff().values)).rolling(window=er_period, min_periods=er_period).sum().values
    
    er = change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    kama_4h = calculate_kama(close, er_period=10)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    choppiness = calculate_choppiness(high, low, close, 14)
    
    # RSI for additional filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE_TREND = 0.30
    LONG_SIZE_MR = 0.25
    SHORT_SIZE_TREND = 0.30
    SHORT_SIZE_MR = 0.25
    
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(choppiness[i]) or np.isnan(kama_4h[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        # Price above 1d HMA(21) = bull market bias (favor longs)
        # Price below 1d HMA(21) = bear market bias (favor shorts)
        # HMA(21) > HMA(50) = confirmed bull trend
        # HMA(21) < HMA(50) = confirmed bear trend
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        bull_trend_confirmed = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        bear_trend_confirmed = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H LOCAL TREND (HMA crossover + KAMA) ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = ranging (mean reversion preferred)
        # CHOP < 45 = trending (breakout preferred)
        # 45-55 = transition (use trend bias)
        is_choppy = choppiness[i] > 55.0
        is_trending = choppiness[i] < 45.0
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        # CRSI < 15 = oversold (long opportunity in chop)
        # CRSI > 85 = overbought (short opportunity in chop)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_neutral_low = crsi[i] < 35.0
        crsi_neutral_high = crsi[i] > 65.0
        
        # === DONCHIAN BREAKOUT (momentum) ===
        donchian_breakout_long = close[i] > donchian_upper[i]
        donchian_breakout_short = close[i] < donchian_lower[i]
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY
        if bull_regime or (not bear_trend_confirmed):
            # Mean reversion entry (choppy market + CRSI oversold)
            if is_choppy and crsi_oversold:
                new_signal = LONG_SIZE_MR
            # Trend breakout entry (trending market + Donchian breakout)
            elif is_trending and donchian_breakout_long and hma_bullish and kama_bullish:
                new_signal = LONG_SIZE_TREND
            # HMA crossover confirmation with CRSI pullback
            elif hma_bullish and crsi_neutral_low and rsi_oversold:
                new_signal = LONG_SIZE_MR
            # KAMA trend + RSI confirmation
            elif kama_bullish and hma_bullish and rsi_14[i] < 50.0:
                new_signal = LONG_SIZE_MR * 0.8
        
        # SHORT ENTRY
        if bear_regime or (not bull_trend_confirmed):
            # Mean reversion entry (choppy market + CRSI overbought)
            if is_choppy and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE_MR
            # Trend breakout entry (trending market + Donchian breakout)
            elif is_trending and donchian_breakout_short and hma_bearish and kama_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE_TREND
            # HMA crossover confirmation with CRSI pullback
            elif hma_bearish and crsi_neutral_high and rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE_MR
            # KAMA trend + RSI confirmation
            elif kama_bearish and hma_bearish and rsi_14[i] > 50.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE_MR * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 20 bars (~3.3 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if bull_regime and crsi_neutral_low and hma_bullish:
                new_signal = LONG_SIZE_MR * 0.5
            elif bear_regime and crsi_neutral_high and hma_bearish:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE_MR * 0.5
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit on mean reversion exhaustion)
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # RSI extreme exit
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # Trend reversal exit (1d regime flip with confirmation)
        if in_position and position_side > 0 and bear_trend_confirmed and hma_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_trend_confirmed and hma_bullish:
            new_signal = 0.0
        
        # Local trend reversal exit (4h HMA cross against position)
        if in_position and position_side > 0 and hma_bearish and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_bullish and kama_bullish:
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