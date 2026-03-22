#!/usr/bin/env python3
"""
Experiment #541: 4h Primary + 1d/1w HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: After 480+ failed strategies, the key insight is REGIME DETECTION.
Most strategies fail because they use trend-following in choppy markets and
mean-reversion in trending markets - the exact wrong approach.

This strategy uses:
1. Choppiness Index (CHOP) to detect market regime:
   - CHOP > 61.8 = choppy/range (use mean reversion)
   - CHOP < 38.2 = trending (use trend following)
   - Between = neutral (reduce position size)
2. Connors RSI (CRSI) for mean reversion entries:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 + price > SMA(200)
   - Short: CRSI > 90 + price < SMA(200)
3. 1d HMA(21) for major trend direction (HTF filter)
4. 1w HMA(50) for secular trend bias
5. ATR(14) 2.5x trailing stop for risk management
6. Dual logic: mean revert in chop, trend follow otherwise

Why this might work:
- Choppiness Index is proven regime filter (research shows 0.8-1.5 Sharpe)
- Connors RSI has 75% win rate for mean reversion
- 4h TF targets 20-50 trades/year (optimal fee/trade ratio)
- Regime switching avoids whipsaws in wrong market conditions
- 1d/1w HTF prevents counter-trend trades in major moves

Position sizing: 0.28 (discrete, max 0.40 per rules)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_connors_1d1w_v2"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile of today's return vs last 100 days
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) - fast RSI for mean reversion
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # Streak RSI - measure consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period):i+1]
        pos_streak = np.sum(streak_vals > 0)
        neg_streak = np.sum(streak_vals < 0)
        total = pos_streak + neg_streak
        if total > 0:
            streak_rsi[i] = 100.0 * pos_streak / total
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank - today's return vs last 100 days
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1] * 100.0
    
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = returns[i-pr_period:i]
        today_return = returns[i]
        percent_rank[i] = 100.0 * np.sum(window < today_return) / pr_period
    
    # Combine into CRSI
    for i in range(pr_period, n):
        crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
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
            chop[i] = 50.0
    
    return chop

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 1w HTF HMA for secular trend
    hma_1w_50 = calculate_hma(df_1w['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    sma_200 = calculate_sma(close, 200)
    
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
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1w_50_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(sma_200[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop_14[i] > 61.8  # Range/mean reversion market
        trending_regime = chop_14[i] < 38.2  # Trending market
        neutral_regime = not choppy_regime and not trending_regime
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 1W SECULAR TREND (bias filter) ===
        secular_bull = close[i] > hma_1w_50_aligned[i]
        secular_bear = close[i] < hma_1w_50_aligned[i]
        
        # === SMA(200) FILTER (long-term trend) ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES (mean reversion signals) ===
        crsi_oversold = crsi[i] < 15.0  # Extreme oversold
        crsi_overbought = crsi[i] > 85.0  # Extreme overbought
        crsi_very_oversold = crsi[i] < 10.0
        crsi_very_overbought = crsi[i] > 90.0
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        
        if choppy_regime:
            # === MEAN REVERSION MODE (choppy market) ===
            # Long: CRSI very oversold + price above SMA200 (bullish bias)
            if crsi_very_oversold and price_above_sma200:
                new_signal = POSITION_SIZE
            # Long: CRSI oversold + bull regime
            elif crsi_oversold and bull_regime:
                new_signal = POSITION_SIZE * 0.8
            # Short: CRSI very overbought + price below SMA200 (bearish bias)
            elif crsi_very_overbought and price_below_sma200:
                new_signal = -POSITION_SIZE
            # Short: CRSI overbought + bear regime
            elif crsi_overbought and bear_regime:
                new_signal = -POSITION_SIZE * 0.8
        
        elif trending_regime:
            # === TREND FOLLOWING MODE (trending market) ===
            # Long: Bull regime + HMA slope bull + CRSI not overbought
            if bull_regime and hma_slope_bull and crsi[i] < 70.0:
                new_signal = POSITION_SIZE
            # Long: Bull regime + secular bull + CRSI pullback
            elif bull_regime and secular_bull and crsi[i] < 50.0:
                new_signal = POSITION_SIZE * 0.8
            # Short: Bear regime + HMA slope bear + CRSI not oversold
            elif bear_regime and hma_slope_bear and crsi[i] > 30.0:
                new_signal = -POSITION_SIZE
            # Short: Bear regime + secular bear + CRSI bounce
            elif bear_regime and secular_bear and crsi[i] > 50.0:
                new_signal = -POSITION_SIZE * 0.8
        
        else:
            # === NEUTRAL MODE (reduce size, wait for clarity) ===
            # Only enter on extreme CRSI with HTF confirmation
            if crsi_very_oversold and bull_regime and secular_bull:
                new_signal = HALF_SIZE
            elif crsi_very_overbought and bear_regime and secular_bear:
                new_signal = -HALF_SIZE
        
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
        
        # === EXIT CONDITIONS (regime flip or extreme CRSI) ===
        # Exit long on regime flip to bear or CRSI overbought
        if in_position and position_side > 0:
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
            elif crsi[i] > 80.0:  # Take profit on mean reversion extreme
                new_signal = 0.0
        
        # Exit short on regime flip to bull or CRSI oversold
        if in_position and position_side < 0:
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
            elif crsi[i] < 20.0:  # Take profit on mean reversion extreme
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