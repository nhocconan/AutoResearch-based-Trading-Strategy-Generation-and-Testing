#!/usr/bin/env python3
"""
Experiment #559: 4h Primary + 1d HTF — Dual Regime Strategy (Choppiness + Connors RSI)

Hypothesis: After analyzing failed strategies (#550, #552, #558 with 0 trades), the pattern is:
- Too many confluence filters = 0 trades (session + volume + chop + ADX all together)
- Pure trend following fails in bear/range markets (2022 crash, 2025 bear)
- Dual regime switching works: trend-follow when CHOP<38.2, mean-revert when CHOP>61.8
- 4h timeframe with 1d HTF = proven combination (20-50 trades/year target)
- Connors RSI for mean-reversion entries has 75% win rate in research
- Choppiness Index is best meta-filter for regime detection

Strategy Logic:
1. 1d HMA(21) for major trend direction (HTF bias)
2. Choppiness Index(14) on 4h: CHOP>61.8 = range, CHOP<38.2 = trend
3. TREND REGIME (CHOP<38.2): Follow 1d trend, enter on 4h RSI(14) pullback (30-50 long, 50-70 short)
4. RANGE REGIME (CHOP>61.8): Mean revert at BB(20,2.5) bounds + Connors RSI extremes
5. ATR(14) 2.5x trailing stop for all positions
6. Position size: 0.28 discrete (balanced for 4h TF)

Why this might beat Sharpe=0.435:
- Dual regime adapts to market conditions (trend vs range)
- 1d HTF prevents major counter-trend losses
- Choppiness filter is proven edge for ETH (Sharpe +0.923 in research)
- Connors RSI catches reversals in bear rallies
- 4h TF = optimal trade frequency (20-50/year, not 200+ fee drag)

Position sizing: 0.28 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_connors_1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with configurable std dev."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 10 (oversold)
    Short: CRSI > 90 (overbought)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI: consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_s = pd.Series(streak)
    streak_gain = streak_s.diff().where(streak_s.diff() > 0, 0.0)
    streak_loss = -streak_s.diff().where(streak_s.diff() < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank: percentage of closes lower than current in lookback
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[i-rank_period+1:i+1]
        count_lower = np.sum(lookback[:-1] < close[i])
        percent_rank[i] = (count_lower / (rank_period - 1)) * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3 + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 61.8 = ranging (mean revert)
        # CHOP < 38.2 = trending (trend follow)
        # 38.2 - 61.8 = transition (no trades or reduced size)
        range_regime = chop_14[i] > 61.8
        trend_regime = chop_14[i] < 38.2
        transition_regime = not range_regime and not trend_regime
        
        new_signal = 0.0
        
        # === TREND REGIME: Follow 1d trend with 4h RSI pullback ===
        if trend_regime:
            # RSI pullback long: 30-50 in uptrend
            rsi_pullback_long = 30.0 <= rsi_14[i] <= 50.0
            # RSI pullback short: 50-70 in downtrend
            rsi_pullback_short = 50.0 <= rsi_14[i] <= 70.0
            
            # LONG: 1d bull + RSI pullback
            if bull_regime_4h := (close[i] > hma_1d_21_aligned[i]) and rsi_pullback_long:
                if hma_1d_slope_bull:
                    new_signal = POSITION_SIZE
                else:
                    new_signal = POSITION_SIZE * 0.7
            
            # SHORT: 1d bear + RSI pullback
            elif bear_regime_1d and rsi_pullback_short:
                if hma_1d_slope_bear:
                    new_signal = -POSITION_SIZE
                else:
                    new_signal = -POSITION_SIZE * 0.7
        
        # === RANGE REGIME: Mean revert at BB bounds + Connors RSI ===
        elif range_regime:
            # Connors RSI extremes for mean reversion
            crsi_oversold = crsi[i] < 15.0
            crsi_overbought = crsi[i] > 85.0
            
            # Price at BB bounds
            at_lower_bb = close[i] <= bb_lower[i] * 1.002  # within 0.2%
            at_upper_bb = close[i] >= bb_upper[i] * 0.998  # within 0.2%
            
            # LONG: CRSI oversold + at lower BB
            if crsi_oversold and at_lower_bb:
                new_signal = POSITION_SIZE * 0.8
            
            # SHORT: CRSI overbought + at upper BB
            elif crsi_overbought and at_upper_bb:
                new_signal = -POSITION_SIZE * 0.8
        
        # === TRANSITION REGIME: Reduce position or flat ===
        elif transition_regime:
            # Only hold existing positions, no new entries
            if in_position:
                new_signal = signals[i-1] if i > 0 else 0.0
            else:
                new_signal = 0.0
        
        # === HOLD POSITION LOGIC ===
        # If already in position and no new signal, maintain
        if in_position and new_signal == 0.0 and not transition_regime:
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
        
        # === EXIT CONDITIONS (regime flip against position) ===
        # Exit long on 1d regime flip to bear in trend regime
        if in_position and position_side > 0 and trend_regime:
            if bear_regime_1d and hma_1d_slope_bear:
                new_signal = 0.0
        
        # Exit short on 1d regime flip to bull in trend regime
        if in_position and position_side < 0 and trend_regime:
            if bull_regime_1d and hma_1d_slope_bull:
                new_signal = 0.0
        
        # Exit range positions when price crosses BB mid
        if in_position and range_regime:
            if position_side > 0 and close[i] > bb_mid[i]:
                new_signal = 0.0  # take profit at mean
            elif position_side < 0 and close[i] < bb_mid[i]:
                new_signal = 0.0  # take profit at mean
        
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