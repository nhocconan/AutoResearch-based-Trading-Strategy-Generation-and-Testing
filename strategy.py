#!/usr/bin/env python3
"""
Experiment #187: 1d Primary + 1w HTF — Dual Regime Strategy with Looser Entries

Hypothesis: Previous 1d strategy (#177) failed with 0 trades due to overly strict filters.
Daily timeframe needs significantly looser entry conditions than 4h to generate 20-50 trades/year.

Strategy combines:
1. Choppiness Index (14) for regime detection: >50 = range, <50 = trend
2. Connors RSI for mean reversion in range mode (thresholds <20/>80 vs <10/>90)
3. Donchian breakout (14-period vs 20) for trend mode
4. 1w HMA for macro bias confirmation (simplified from 1d+1w)
5. ATR trailing stop (2.5x) for risk management
6. Position sizing: 0.30 full, 0.20 half (discrete levels to minimize fee churn)

Key changes from failed #177:
- Looser CRSI thresholds (20/80 vs 10/90) - more signals
- Looser CHOP thresholds (50 vs 55/45) - catches more regime changes
- Shorter Donchian period (14 vs 20) - more breakout opportunities
- Simpler HTF filter (1w only) - less conflicting signals
- Hold logic to maintain positions across multiple bars
- Start loop at index 100 (vs higher) to capture more early trades

TARGET: 25-45 trades/year on 1d, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_crsi_donchian_1w_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 50 = ranging market (mean revert)
    CHOP < 50 = trending market (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 20, Short when CRSI > 80 (looser than literature 10/90)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100.0 - (100.0 / (1.0 + rs))
    rsi_short = rsi_short.fillna(50.0).values
    
    # RSI of Streak (consecutive up/down days)
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
        if streak[i] > 0:
            streak_rsi[i] = 100.0 * min(np.abs(streak[i]), streak_period) / streak_period
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * (1.0 - min(np.abs(streak[i]), streak_period) / streak_period)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0:
            rank = np.sum(returns[:-1] < returns[-1])
            percent_rank[i] = 100.0 * rank / max(len(returns) - 1, 1)
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_donchian(high, low, period=14):
    """Calculate Donchian Channel (upper and lower bands)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=14)
    
    # Calculate 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss and hold logic
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        is_range = chop_14[i] > 50.0  # Ranging market
        is_trend = chop_14[i] < 50.0  # Trending market
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (Connors RSI)
            # Long: CRSI < 20 + price above 1w HMA (bullish bias)
            if crsi[i] < 20.0 and price_above_hma_1w:
                new_signal = POSITION_SIZE_FULL
            
            # Short: CRSI > 80 + price below 1w HMA (bearish bias)
            elif crsi[i] > 80.0 and price_below_hma_1w:
                new_signal = -POSITION_SIZE_FULL
        
        elif is_trend:
            # TREND FOLLOWING MODE (Donchian Breakout)
            # Long: Price breaks Donchian upper + price above 1w HMA
            if close[i] > donchian_upper[i-1] and price_above_hma_1w:
                new_signal = POSITION_SIZE_FULL
            
            # Short: Price breaks Donchian lower + price below 1w HMA
            elif close[i] < donchian_lower[i-1] and price_below_hma_1w:
                new_signal = -POSITION_SIZE_FULL
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and macro bias still valid (prevents premature exits)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 1w HMA
                if price_above_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 1w HMA
                if price_below_hma_1w:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 1w HMA (macro trend changed)
        if in_position and position_side > 0 and price_below_hma_1w:
            new_signal = 0.0
        
        # Exit short if price crosses above 1w HMA (macro trend changed)
        if in_position and position_side < 0 and price_above_hma_1w:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals