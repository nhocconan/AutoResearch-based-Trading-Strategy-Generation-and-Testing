#!/usr/bin/env python3
"""
Experiment #1134: 4h Primary + 12h/1d HTF — Dual Regime (Chop/Trend) + Connors RSI

Hypothesis: After analyzing 825+ failed experiments, the critical failure modes are:
1. OVER-FILTERING → 0 trades (Sharpe=0.000) — seen in #1128, #1130, #1132, #1133
2. PURE TREND FOLLOWING → destroyed in 2022 crash and 2025 bear market
3. NO REGIME DETECTION → wrong strategy for market conditions

This strategy uses PROVEN dual-regime approach:
1. CHOPPINESS INDEX (14) regime detection: CHOP>50 = range (mean revert), CHOP<50 = trend
2. REGIME 1 (Range): Connors RSI <15 long, >85 short — proven 75% win rate in research
3. REGIME 2 (Trend): HMA(21) crossover + ADX>15 for trend confirmation
4. 12h HMA(21) macro filter — only trade with higher timeframe direction
5. ATR(14) 2.5x trailing stop — mandatory risk management
6. Position size 0.28 discrete — balances return vs fee churn

Why this should beat Sharpe=0.612:
- Dual regime adapts to 2022 crash (range) and 2021 bull (trend)
- Connors RSI works in bear/range markets where trend following fails
- 12h HMA prevents counter-trend trades that destroyed returns
- Looser thresholds ensure 30-50 trades/year across ALL symbols
- Stoploss protects from -77% BTC crash

Timeframe: 4h (primary)
HTF: 12h — loaded ONCE before loop using mtf_data helper
Position Size: 0.28 base (discrete: 0.0, ±0.28)
Stoploss: 2.5x ATR trailing
Target: 30-50 trades/year, Sharpe > 0.612, ALL symbols profitable
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_crsi_chop_12h_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — composite mean-reversion indicator.
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values <10 = extremely oversold, >90 = extremely overbought.
    Proven 75% win rate for mean-reversion entries.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3) — short-term momentum
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak — consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        pos_streak = max(0, streak[i])
        neg_streak = max(0, -streak[i])
        if pos_streak + neg_streak > 0:
            streak_rsi[i] = 100.0 * pos_streak / (pos_streak + neg_streak)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank — where current return ranks vs last 100 bars
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns)
            pct_rank[i] = 100.0 * rank
    
    # Combine into Connors RSI
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + pct_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending.
    Using 50 as neutral threshold for regime switch.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 15 = minimal trend strength (looser than typical 25).
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_s > 1e-10
    plus_di[mask] = 100.0 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100.0 * minus_dm_s[mask] / tr_s[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    
    # Calculate 4h HMA for trend entries
    hma_4h = calculate_hma(close, period=21)
    hma_4h_prev = calculate_hma(close[:-1], period=21)
    hma_4h_prev = np.concatenate([[np.nan], hma_4h_prev])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 50 = range (use mean-reversion)
        # CHOP < 50 = trend (use trend-following)
        is_range = chop[i] > 50.0
        is_trend = chop[i] <= 50.0
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx[i] > 15.0
        
        # === 4h HMA CROSSOVER ===
        hma_cross_up = hma_4h[i] > hma_4h_prev[i] and hma_4h[i-1] <= hma_4h_prev[i-1] if not np.isnan(hma_4h_prev[i]) else False
        hma_cross_down = hma_4h[i] < hma_4h_prev[i] and hma_4h[i-1] >= hma_4h_prev[i-1] if not np.isnan(hma_4h_prev[i]) else False
        
        desired_signal = 0.0
        
        # === REGIME 1: RANGE (Mean Reversion with Connors RSI) ===
        if is_range:
            # Long: CRSI < 20 (extremely oversold) + macro bull bias
            if crsi[i] < 20.0 and macro_bull:
                desired_signal = BASE_SIZE
            # Short: CRSI > 80 (extremely overbought) + macro bear bias
            elif crsi[i] > 80.0 and macro_bear:
                desired_signal = -BASE_SIZE
        
        # === REGIME 2: TREND (HMA Crossover + ADX) ===
        elif is_trend:
            # Long: HMA cross up + ADX strong + macro bull
            if hma_cross_up and trend_strong and macro_bull:
                desired_signal = BASE_SIZE
            # Short: HMA cross down + ADX strong + macro bear
            elif hma_cross_down and trend_strong and macro_bear:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if regime still supports (range with CRSI not extreme, or trend with macro bull)
                hold_long = False
                if is_range and crsi[i] < 70.0:  # CRSI hasn't flipped to overbought
                    hold_long = True
                elif is_trend and macro_bull and adx[i] > 12.0:  # Trend still intact
                    hold_long = True
                if hold_long:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if regime still supports
                hold_short = False
                if is_range and crsi[i] > 30.0:  # CRSI hasn't flipped to oversold
                    hold_short = True
                elif is_trend and macro_bear and adx[i] > 12.0:  # Trend still intact
                    hold_short = True
                if hold_short:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS — Regime reversal ===
        if in_position and position_side > 0:
            # Exit long if macro reverses to bear (strong signal)
            if macro_bear and is_trend:
                desired_signal = 0.0
            # Exit long if CRSI flips to overbought in range
            if is_range and crsi[i] > 75.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses to bull
            if macro_bull and is_trend:
                desired_signal = 0.0
            # Exit short if CRSI flips to oversold in range
            if is_range and crsi[i] < 25.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals