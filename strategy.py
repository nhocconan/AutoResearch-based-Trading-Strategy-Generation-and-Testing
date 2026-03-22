#!/usr/bin/env python3
"""
Experiment #561: 4h Primary + 1d/1w HTF — Dual Regime (Choppiness + Connors RSI + HMA)

Hypothesis: After #555 failed (1h Sharpe=-0.484), moving to 4h primary with dual-regime logic.
Research shows Connors RSI + Choppiness worked well on ETH (Sharpe +0.923 in backtests).
Key insight: Different market regimes need different strategies.

REGIME DETECTION (Choppiness Index):
- CHOP > 61.8 = Range/Chop → Mean Reversion (Connors RSI extremes)
- CHOP < 38.2 = Trend → Trend Following (HMA crossover + pullback)
- 38.2 <= CHOP <= 61.8 = Transition → Stay flat or reduce size

HTF FILTER (1d HMA):
- Price > 1d HMA(50) = Bullish bias (prefer longs, skip shorts)
- Price < 1d HMA(50) = Bearish bias (prefer shorts, skip longs)

ENTRY LOGIC:
- Range regime: Connors RSI < 15 long, > 85 short (with 1d bias filter)
- Trend regime: HMA(16) > HMA(48) + RSI(14) pullback to 40-55 for long
- Exit: Regime flip, ATR stoploss (2.5x), or HTF bias flip

Why this might beat Sharpe=0.435:
- Dual-regime adapts to market conditions (not one-size-fits-all)
- Connors RSI proven for mean reversion (75% win rate in literature)
- Choppiness filter avoids trend strategies in chop (major source of losses)
- 4h TF = 20-50 trades/year target (per Rule 10)
- 1d HTF prevents major counter-trend positions

Position sizing: 0.28 base (discrete, max 0.40 per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_connors_hma_1d1w_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = Range/Chop
    CHOP < 38.2 = Trend
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Very short-term momentum
    RSI(Streak): Consecutive up/down days
    PercentRank: Where current price ranks vs last 100 days
    """
    n = len(close)
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
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: Percent Rank
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_close.values + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF HMA for major trend bias
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_200 = calculate_hma(df_1d['close'].values, period=200)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1d_200_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_200)
    
    # Calculate 1w HTF HMA for very long-term bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, 14)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_50_aligned[i]) or np.isnan(hma_1d_200_aligned[i]):
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            continue
        
        # === 1D MAJOR TREND BIAS (primary direction filter) ===
        bull_bias_1d = close[i] > hma_1d_50_aligned[i]
        bear_bias_1d = close[i] < hma_1d_50_aligned[i]
        
        # Strong bias confirmation
        strong_bull_1d = bull_bias_1d and hma_1d_50_aligned[i] > hma_1d_200_aligned[i]
        strong_bear_1d = bear_bias_1d and hma_1d_50_aligned[i] < hma_1d_200_aligned[i]
        
        # === 1W VERY LONG-TERM BIAS ===
        bull_bias_1w = close[i] > hma_1w_21_aligned[i]
        bear_bias_1w = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        chop_value = chop_14[i]
        range_regime = chop_value > 61.8  # Mean reversion territory
        trend_regime = chop_value < 38.2  # Trend following territory
        transition_regime = 38.2 <= chop_value <= 61.8  # Unclear
        
        # === CONNORS RSI FOR MEAN REVERSION ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought
        crsi_mild_oversold = 15.0 <= crsi[i] < 30.0
        crsi_mild_overbought = 70.0 < crsi[i] <= 85.0
        
        # === HMA TREND FOR TREND FOLLOWING ===
        hma_bull_trend = hma_16[i] > hma_48[i]
        hma_bear_trend = hma_16[i] < hma_48[i]
        
        # RSI pullback in trend
        rsi_pullback_long = 40.0 <= rsi_14[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_14[i] <= 60.0
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # RANGE REGIME: Mean Reversion with Connors RSI
        if range_regime:
            # Long: CRSI oversold + 1d bull bias (or neutral)
            if crsi_oversold and (bull_bias_1d or not strong_bear_1d):
                new_signal = POSITION_SIZE
            elif crsi_mild_oversold and strong_bull_1d:
                new_signal = HALF_SIZE
            
            # Short: CRSI overbought + 1d bear bias (or neutral)
            elif crsi_overbought and (bear_bias_1d or not strong_bull_1d):
                new_signal = -POSITION_SIZE
            elif crsi_mild_overbought and strong_bear_1d:
                new_signal = -HALF_SIZE
        
        # TREND REGIME: Trend Following with HMA + RSI pullback
        elif trend_regime:
            # Long: HMA bull + RSI pullback + 1d bull bias
            if hma_bull_trend and rsi_pullback_long and bull_bias_1d:
                if strong_bull_1d and bull_bias_1w:
                    new_signal = POSITION_SIZE
                else:
                    new_signal = HALF_SIZE
            
            # Short: HMA bear + RSI pullback + 1d bear bias
            elif hma_bear_trend and rsi_pullback_short and bear_bias_1d:
                if strong_bear_1d and bear_bias_1w:
                    new_signal = -POSITION_SIZE
                else:
                    new_signal = -HALF_SIZE
        
        # TRANSITION REGIME: Stay flat or hold existing
        # (no new entries, but can hold existing positions)
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === EXIT CONDITIONS (regime flip or bias flip) ===
        # Exit long on strong bear bias flip
        if in_position and position_side > 0:
            if strong_bear_1d and bear_bias_1w:
                new_signal = 0.0
            # Exit in transition regime if CRSI moves against
            if transition_regime and crsi[i] > 70.0:
                new_signal = 0.0
        
        # Exit short on strong bull bias flip
        if in_position and position_side < 0:
            if strong_bull_1d and bull_bias_1w:
                new_signal = 0.0
            # Exit in transition regime if CRSI moves against
            if transition_regime and crsi[i] < 30.0:
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