#!/usr/bin/env python3
"""
Experiment #722: 12h Primary + 1d/1w HTF — Choppiness Regime + Connors RSI

Hypothesis: After 482 failed strategies, the pattern is clear:
- Simple trend following fails in bear/range markets (2022 crash, 2025 bear)
- Pure mean reversion fails in strong trends
- SOLUTION: Regime-adaptive strategy using Choppiness Index

This strategy uses:
1. Choppiness Index (CHOP) to detect range vs trend regime
2. Connors RSI (CRSI) for mean reversion entries in range regimes
3. HMA trend + RSI pullback for trend-following entries in trend regimes
4. 1d HMA for primary trend bias, 1w HMA for ultra-long-term filter
5. ATR trailing stoploss (2.5x) for risk management

Key design choices:
- CHOP > 61.8 = range regime → use Connors RSI mean reversion
- CHOP < 38.2 = trend regime → use HMA + RSI pullback trend following
- Multiple entry paths to ensure trade frequency (critical for 12h TF)
- Position size 0.25-0.30 (discrete levels to minimize fee churn)
- Target: 20-50 trades/year at 12h timeframe

Timeframe: 12h (proven higher TF works best, lower fee drag)
Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_crsi_hma_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback period
    
    CRSI < 10 = deeply oversold (long signal)
    CRSI > 90 = deeply overbought (short signal)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - consecutive up/down days
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak values
    streak_delta = np.diff(streak)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    avg_streak_gain = np.concatenate([[np.nan], avg_streak_gain])
    avg_streak_loss = np.concatenate([[np.nan], avg_streak_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        streak_rsi = 100 - (100 / (1 + streak_rs))
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank - percentile of price change over lookback
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        price_changes = np.diff(close[i-rank_period:i+1])
        if len(price_changes) > 0 and not np.all(np.isnan(price_changes)):
            current_change = price_changes[-1]
            valid_changes = price_changes[~np.isnan(price_changes)]
            if len(valid_changes) > 0:
                pct_rank[i] = np.sum(valid_changes <= current_change) / len(valid_changes) * 100
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(pct_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + pct_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 5:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop_raw, 0, 100)
    return chop

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel for breakout detection."""
    n = len(high)
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
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(350, n):  # Need buffer for all indicators + HTF alignment + CRSI
        # Skip if indicators not ready
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(chop_12h[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(donch_upper[i]) or np.isnan(crsi_12h[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range/choppy (mean reversion)
        # CHOP < 38.2 = trending (trend following)
        # 38.2 <= CHOP <= 61.8 = neutral (use trend bias)
        is_range_regime = chop_12h[i] > 61.8
        is_trend_regime = chop_12h[i] < 38.2
        
        # === TREND BIAS (1d and 1w HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong trend when both 1d and 1w agree
        strong_bullish = trend_1d_bullish and trend_1w_bullish
        strong_bearish = trend_1d_bearish and trend_1w_bearish
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY CONDITIONS ===
        long_signal = False
        
        # RANGE REGIME: Connors RSI mean reversion (CRSI < 15 = deeply oversold)
        if is_range_regime:
            if crsi_12h[i] < 15 and above_sma200:
                long_signal = True
            # Additional range entry: CRSI very low + price near Donchian lower
            if crsi_12h[i] < 25 and close[i] < donch_lower[i-1] * 1.02:
                long_signal = True
        
        # TREND REGIME: HMA + RSI pullback entries
        if is_trend_regime or not is_range_regime:
            # Path 1: Strong bullish trend + RSI pullback
            if strong_bullish and rsi_12h[i] < 50 and above_sma200:
                long_signal = True
            
            # Path 2: Bullish trend + Donchian breakout
            if trend_1d_bullish and close[i] > donch_upper[i-1] and rsi_12h[i] < 65:
                long_signal = True
            
            # Path 3: RSI deeply oversold in uptrend (mean reversion within trend)
            if rsi_12h[i] < 35 and trend_1d_bullish:
                long_signal = True
            
            # Path 4: CRSI oversold + bullish trend (combined signal)
            if crsi_12h[i] < 30 and trend_1d_bullish and above_sma200:
                long_signal = True
        
        # NEUTRAL REGIME: Use 1w HMA for bias + CRSI for timing
        if not is_range_regime and not is_trend_regime:
            if trend_1w_bullish and crsi_12h[i] < 25:
                long_signal = True
            if trend_1w_bullish and rsi_12h[i] < 40 and above_sma200:
                long_signal = True
        
        if long_signal:
            desired_signal = current_size
        
        # === SHORT ENTRY CONDITIONS ===
        short_signal = False
        
        # RANGE REGIME: Connors RSI mean reversion (CRSI > 85 = deeply overbought)
        if is_range_regime:
            if crsi_12h[i] > 85 and below_sma200:
                short_signal = True
            # Additional range entry: CRSI very high + price near Donchian upper
            if crsi_12h[i] > 75 and close[i] > donch_upper[i-1] * 0.98:
                short_signal = True
        
        # TREND REGIME: HMA + RSI bounce entries
        if is_trend_regime or not is_range_regime:
            # Path 1: Strong bearish trend + RSI bounce
            if strong_bearish and rsi_12h[i] > 50 and below_sma200:
                short_signal = True
            
            # Path 2: Bearish trend + Donchian breakdown
            if trend_1d_bearish and close[i] < donch_lower[i-1] and rsi_12h[i] > 35:
                short_signal = True
            
            # Path 3: RSI deeply overbought in downtrend
            if rsi_12h[i] > 65 and trend_1d_bearish:
                short_signal = True
            
            # Path 4: CRSI overbought + bearish trend
            if crsi_12h[i] > 70 and trend_1d_bearish and below_sma200:
                short_signal = True
        
        # NEUTRAL REGIME: Use 1w HMA for bias + CRSI for timing
        if not is_range_regime and not is_trend_regime:
            if trend_1w_bearish and crsi_12h[i] > 75:
                short_signal = True
            if trend_1w_bearish and rsi_12h[i] > 60 and below_sma200:
                short_signal = True
        
        if short_signal:
            desired_signal = -current_size
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with stronger trend (1w HMA)
        if long_signal and short_signal:
            if trend_1w_bullish:
                desired_signal = current_size
            elif trend_1w_bearish:
                desired_signal = -current_size
            else:
                desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if regime/trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend_1d still bullish or range regime with CRSI not extreme
                if trend_1d_bullish or (is_range_regime and crsi_12h[i] < 70):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend_1d still bearish or range regime with CRSI not extreme
                if trend_1d_bearish or (is_range_regime and crsi_12h[i] > 30):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI/RSI extremely overbought
            if trend_1d_bearish or crsi_12h[i] > 85 or rsi_12h[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI/RSI extremely oversold
            if trend_1d_bullish or crsi_12h[i] < 15 or rsi_12h[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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