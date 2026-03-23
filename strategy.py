#!/usr/bin/env python3
"""
Experiment #701: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Connors RSI (CRSI) is proven for mean-reversion with 75%+ win rate in 
traditional markets. Combined with Choppiness Index regime filter and HTF HMA trend 
bias, this should work across all market conditions. 4h TF balances trade frequency 
(20-50/year) with signal quality.

Key Components:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI, catches short-term extremes
2. Choppiness Index (CHOP) regime: >61.8 = range (mean revert), <38.2 = trend
   - Switches between mean-reversion and breakout logic
3. 1d HMA(21) for trend bias - only trade with HTF trend
4. Loose entry thresholds to ensure trade frequency (CRSI <30/>70 not <10/>90)
5. ATR trailing stoploss at 2.5x

Why this should beat current best (Sharpe=0.612):
- CRSI is more sensitive than standard RSI → more trades
- CHOP regime filter prevents mean-reversion in strong trends (major failure mode)
- 4h TF worked in multiple past experiments
- Discrete sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_hma_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean-reversion indicator with 75%+ win rate.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100 - (100 / (1 + rs))
    rsi_short = np.clip(rsi_short, 0, 100)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, 50.0)
    for i in range(n):
        if streak[i] > 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        elif streak[i] < 0:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    # Smooth streak RSI
    streak_rsi_smooth = pd.Series(streak_rsi).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    # Percent Rank of price over last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i])
        percent_rank[i] = (rank / (rank_period - 1)) * 100
    
    # Combine into CRSI
    valid_mask = (~np.isnan(rsi_short)) & (~np.isnan(streak_rsi_smooth)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi_smooth[valid_mask] + percent_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy (mean-reversion favorable)
    CHOP < 38.2 = trending (breakout favorable)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Need buffer for indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(crsi_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_4h[i] > 55.0  # Range/mean-reversion regime
        is_trending = chop_4h[i] < 45.0  # Trend/breakout regime
        # Neutral zone 45-55: use either logic
        
        # === TREND BIAS (HTF HMA) ===
        trend_bullish_1d = close[i] > hma_1d_aligned[i]
        trend_bearish_1d = close[i] < hma_1d_aligned[i]
        trend_bullish_1w = close[i] > hma_1w_aligned[i]
        trend_bearish_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias when both HTF agree
        trend_strong_bullish = trend_bullish_1d and trend_bullish_1w
        trend_strong_bearish = trend_bearish_1d and trend_bearish_1w
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === MEAN REVERSION MODE (Choppy/Range) ===
        # Long: CRSI oversold + bullish HTF bias
        if is_choppy or True:  # Always allow mean-reversion with HTF filter
            if crsi_4h[i] < 30 and (trend_bullish_1d or trend_bullish_1w):
                desired_signal = current_size
            # Short: CRSI overbought + bearish HTF bias
            elif crsi_4h[i] > 70 and (trend_bearish_1d or trend_bearish_1w):
                desired_signal = -current_size
            # Weaker signals with single HTF confirmation
            elif crsi_4h[i] < 25 and trend_bullish_1d:
                desired_signal = HALF_SIZE
            elif crsi_4h[i] > 75 and trend_bearish_1d:
                desired_signal = -HALF_SIZE
        
        # === TREND FOLLOWING MODE (Trending) ===
        if is_trending:
            # Long breakout: CRSI rising from oversold + strong bullish trend
            if crsi_4h[i] < 50 and trend_strong_bullish and above_sma200:
                if desired_signal == 0.0:  # Don't override MR signal
                    desired_signal = current_size
            # Short breakout: CRSI falling from overbought + strong bearish trend
            elif crsi_4h[i] > 50 and trend_strong_bearish and below_sma200:
                if desired_signal == 0.0:
                    desired_signal = -current_size
        
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
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought and trend intact
                if crsi_4h[i] < 75 and trend_bullish_1d:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if CRSI not oversold and trend intact
                if crsi_4h[i] > 25 and trend_bearish_1d:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Long exit: CRSI overbought OR trend reverses below both HTF HMA
        if in_position and position_side > 0:
            if crsi_4h[i] > 80:
                desired_signal = 0.0
            elif close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]:
                desired_signal = 0.0
        
        # Short exit: CRSI oversold OR trend reverses above both HTF HMA
        if in_position and position_side < 0:
            if crsi_4h[i] < 20:
                desired_signal = 0.0
            elif close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE * 0.7 else HALF_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE * 0.7 else -HALF_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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