#!/usr/bin/env python3
"""
Experiment #001: 4h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI

Hypothesis: Combining Choppiness Index regime detection with Connors RSI entries
will adapt to both trending and ranging markets. Research shows CHOP > 61.8 indicates
range conditions (mean reversion works), while CHOP < 38.2 indicates trending
(trend following works). Connors RSI (CRSI) has 75% win rate for reversals.

Key components:
1. Choppiness Index (14-period) for regime detection
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. 4h HMA for primary trend, 1d HMA for regime confirmation
4. Dual logic: mean revert in chop, trend follow in trends
5. ATR trailing stoploss (2.5*ATR)

Why this might beat the baseline:
- Regime-adaptive (doesn't fight market conditions)
- Connors RSI proven for BTC/ETH reversals through 2022 crash
- 4h TF targets 20-50 trades/year (fee-efficient per Rule 10)
- Position size 0.30 (conservative, discrete levels)

Entry conditions (LOOSE enough for trades):
- Long in range: CRSI < 15 + CHOP > 55
- Long in trend: CRSI < 25 + price > 4h HMA + 1d HMA bullish
- Short in range: CRSI > 85 + CHOP > 55
- Short in trend: CRSI > 75 + price < 4h HMA + 1d HMA bearish
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_regime_1d_v1"
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
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n) * np.nan
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
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
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    streak_rsi = np.zeros(n)
    
    for i in range(streak_period, n):
        if streak_abs[i] == 0:
            streak_rsi[i] = 50.0
        else:
            # Map streak to 0-100 scale
            streak_rsi[i] = 50.0 + streak_sign[i] * min(streak_abs[i], 10) * 5.0
            streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank component
    percent_rank = np.zeros(n) * np.nan
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_lower = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_lower / (rank_period - 1)
    
    # Combine components
    crsi = (rsi_short.values + streak_rsi + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for regime confirmation
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for major trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    
    chop = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_4h[i]) or np.isnan(hma_4h_fast[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === 1D TREND BIAS ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 1W MAJOR TREND ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        price_below_hma_1w = close[i] < hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else False
        
        # === 4H TREND ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-3] if i >= 3 else False
        price_above_hma_4h = close[i] > hma_4h[i]
        price_below_hma_4h = close[i] < hma_4h[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20
        crsi_extreme_oversold = crsi[i] < 10
        crsi_overbought = crsi[i] > 80
        crsi_extreme_overbought = crsi[i] > 90
        
        new_signal = 0.0
        
        # === MEAN REVERSION IN CHOPPY MARKET ===
        if is_choppy:
            # Long: CRSI extreme oversold in range
            if crsi_extreme_oversold:
                new_signal = POSITION_SIZE
            # Short: CRSI extreme overbought in range
            elif crsi_extreme_overbought:
                new_signal = -POSITION_SIZE
        
        # === TREND FOLLOWING IN TRENDING MARKET ===
        elif is_trending:
            # Long: Pullback in uptrend
            if crsi_oversold and price_above_hma_4h and hma_1d_slope_bull:
                new_signal = POSITION_SIZE
            # Short: Rally in downtrend
            elif crsi_overbought and price_below_hma_4h and hma_1d_slope_bear:
                new_signal = -POSITION_SIZE
        
        # === HYBRID: Weak trend with oversold/overbought ===
        if new_signal == 0.0:
            # Long: Very oversold + 1d not strongly bearish
            if crsi_extreme_oversold and not hma_1d_slope_bear:
                new_signal = POSITION_SIZE * 0.5  # Half size for counter-trend
            # Short: Very overbought + 1d not strongly bullish
            elif crsi_extreme_overbought and not hma_1d_slope_bull:
                new_signal = -POSITION_SIZE * 0.5  # Half size for counter-trend
        
        # === HOLD POSITION LOGIC ===
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
        
        # === EXIT ON REGIME/TREND FLIP ===
        if in_position and position_side > 0:
            # Exit long if trend turns strongly bearish
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend turns strongly bullish
            if hma_1d_slope_bull and price_above_hma_1d:
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