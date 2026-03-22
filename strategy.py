#!/usr/bin/env python3
"""
Experiment #522: 12h Primary + 1d/1w HTF — Donchian Breakout + HMA Trend + Choppiness Regime

Hypothesis: After 467 failed strategies, higher timeframes (12h) with simpler logic work best.
This combines proven patterns from research notes:
- Donchian breakout + HMA trend + RSI (SOL Sharpe +0.782)
- Choppiness Index regime switch (ETH Sharpe +0.923)
- HMA crossover + RSI filter + ATR trail (SOL +0.879)

Key design decisions:
1. 12h primary TF targets 20-50 trades/year (optimal fee/trade ratio for this TF)
2. 1d HMA(21) for major trend direction - only trade with HTF trend
3. 1w HMA(50) for ultra-long-term regime (bull/bear market filter)
4. Choppiness Index(14) to detect range vs trend regime
5. Donchian(20) breakout for entry timing - catches momentum moves
6. Connors RSI for pullback entries in trending regime
7. ATR(14) 2.5x trailing stop for risk management

Why this might work:
- 12h TF has proven success (research notes show multiple 12h strategies with Sharpe > 0.7)
- Donchian breakouts catch sustained moves without whipsaw
- Choppiness filter avoids trend strategies in range markets (major failure mode)
- 1w regime filter prevents counter-trend trades in strong bear markets (2022 crash)
- Simpler logic than volspike/Fisher combos = consistent signals across BTC/ETH/SOL
- Discrete position sizing (0.25-0.30) minimizes fee churn

Position sizing: 0.25-0.30 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_chop_regime_1d1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) - mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Research shows 75% win rate for CRSI < 10 (long) and CRSI > 90 (short).
    """
    close_s = pd.Series(close)
    
    # RSI(3) component
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI component
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank component
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection.
    CHOP > 61.8 = range/choppy market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    tr_s = pd.Series(tr)
    atr_sum = tr_s.rolling(window=period, min_periods=period).sum()
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    price_range = highest_high - lowest_low
    
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    return chop.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 1w HTF indicators
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Donchian breakouts
    prev_donchian_upper = np.zeros(n)
    prev_donchian_lower = np.zeros(n)
    prev_donchian_upper[1:] = donchian_upper[:-1]
    prev_donchian_lower[1:] = donchian_lower[:-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(hma_1w_50_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        # === 1W ULTRA-LONG-TERM REGIME ===
        bull_market = close[i] > hma_1w_50_aligned[i]
        bear_market = close[i] < hma_1w_50_aligned[i]
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        trending_regime = chop[i] < 45.0  # Trending market
        ranging_regime = chop[i] > 55.0  # Range/choppy market
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_up = (close[i] > donchian_upper[i]) and (close[i-1] <= prev_donchian_upper[i])
        donchian_breakout_down = (close[i] < donchian_lower[i]) and (close[i-1] >= prev_donchian_lower[i])
        
        # Donchian position (price relative to channel)
        donchian_mid = (donchian_upper[i] + donchian_lower[i]) / 2.0
        price_in_upper_half = close[i] > donchian_mid
        price_in_lower_half = close[i] < donchian_mid
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        crsi_oversold = crsi[i] < 15.0  # Strong long signal
        crsi_overbought = crsi[i] > 85.0  # Strong short signal
        crsi_neutral_long = crsi[i] < 50.0
        crsi_neutral_short = crsi[i] > 50.0
        
        # === RSI FILTER ===
        rsi_neutral_long = rsi_14[i] < 70.0
        rsi_neutral_short = rsi_14[i] > 30.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC — REGIME-ADAPTIVE ===
        new_signal = 0.0
        
        # TRENDING REGIME: Follow breakouts with trend
        if trending_regime:
            # LONG: Donchian breakout up + bull regime + RSI not overbought
            if donchian_breakout_up and bull_regime and rsi_neutral_long:
                new_signal = POSITION_SIZE
            # LONG: Bull regime + CRSI oversold (pullback in uptrend)
            elif bull_regime and crsi_oversold and hma_slope_bull:
                new_signal = POSITION_SIZE * 0.9
            # LONG: Bull market (1w) + bull regime + price in upper Donchian
            elif bull_market and bull_regime and price_in_upper_half:
                new_signal = POSITION_SIZE * 0.8
            
            # SHORT: Donchian breakout down + bear regime + RSI not oversold
            if new_signal == 0.0:
                if donchian_breakout_down and bear_regime and rsi_neutral_short:
                    new_signal = -POSITION_SIZE
                # SHORT: Bear regime + CRSI overbought (bounce in downtrend)
                elif bear_regime and crsi_overbought and hma_slope_bear:
                    new_signal = -POSITION_SIZE * 0.9
                # SHORT: Bear market (1w) + bear regime + price in lower Donchian
                elif bear_market and bear_regime and price_in_lower_half:
                    new_signal = -POSITION_SIZE * 0.8
        
        # RANGING REGIME: Mean reversion at channel bounds
        elif ranging_regime:
            # LONG: CRSI oversold + price at Donchian lower + RSI oversold
            if crsi_oversold and close[i] <= donchian_lower[i] * 1.02 and rsi_oversold:
                new_signal = POSITION_SIZE * 0.9
            # LONG: Price at Donchian lower + bull regime (support hold)
            elif close[i] <= donchian_lower[i] * 1.02 and bull_regime:
                new_signal = POSITION_SIZE * 0.7
            
            # SHORT: CRSI overbought + price at Donchian upper + RSI overbought
            if new_signal == 0.0:
                if crsi_overbought and close[i] >= donchian_upper[i] * 0.98 and rsi_overbought:
                    new_signal = -POSITION_SIZE * 0.9
                # SHORT: Price at Donchian upper + bear regime (resistance hold)
                elif close[i] >= donchian_upper[i] * 0.98 and bear_regime:
                    new_signal = -POSITION_SIZE * 0.7
        
        # NEUTRAL REGIME: Wait for clear signals
        else:
            # Only take strongest signals in neutral regime
            if donchian_breakout_up and bull_regime and crsi_oversold:
                new_signal = POSITION_SIZE * 0.8
            elif donchian_breakout_down and bear_regime and crsi_overbought:
                new_signal = -POSITION_SIZE * 0.8
        
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
        
        # === EXIT CONDITIONS (regime flip or extreme) ===
        if in_position and position_side > 0:
            # Exit long on regime flip to strong bear
            if bear_regime and hma_slope_bear and bear_market:
                new_signal = 0.0
            # Exit on extreme overbought
            elif rsi_14[i] > 80.0 or crsi[i] > 90.0:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short on regime flip to strong bull
            if bull_regime and hma_slope_bull and bull_market:
                new_signal = 0.0
            # Exit on extreme oversold
            elif rsi_14[i] < 20.0 or crsi[i] < 10.0:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals