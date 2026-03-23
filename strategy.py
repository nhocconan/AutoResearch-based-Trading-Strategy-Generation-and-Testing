#!/usr/bin/env python3
"""
Experiment #175: 1h Primary + 4h/1d HTF — Connors RSI + HMA Trend + Choppiness Regime

Hypothesis: Previous 1h strategies failed due to overly strict entry conditions (0 trades).
Connors RSI (CRSI) has proven 75% win rate for mean reversion in bear/range markets.
Combined with 4h HMA trend filter and Choppiness regime detection, this should generate
consistent trades across ALL symbols while maintaining positive Sharpe.

KEY IMPROVEMENTS over #172:
1. Connors RSI (RSI3 + RSI_Streak + PercentRank) / 3 — proven mean reversion signal
2. Simpler entry: CRSI < 15 (long) or > 85 (short) + HTF trend alignment
3. Looser confluence: only 2 factors required (not 3+)
4. Remove session filter (crypto trades 24/7)
5. Position size: 0.25 full, 0.15 partial (discrete levels)
6. ATR trailing stop at 2.5x for risk management

TARGET: 40-70 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_chop_4h1d_v1"
timeframe = "1h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Fast RSI for short-term momentum
    RSI(Streak, 2): RSI of consecutive up/down days
    PercentRank(100): Percentile rank of price change over 100 periods
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - fast RSI
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close = rsi_close.fillna(50.0).values
    
    # RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100) - percentile rank of price change
    pct_change = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = pct_change[i-rank_period+1:i+1]
        current = pct_change[i]
        if len(window) > 0:
            percent_rank[i] = np.sum(window < current) / len(window) * 100.0
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    with np.errstate(divide='ignore', invalid='ignore'):
        crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = rsi.fillna(50.0).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Calculate 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    POSITION_SIZE_QUARTER = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === VOLUME FILTER (lenient for 1h) ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === REGIME DETECTION ===
        chop_value = chop_14[i]
        is_trending = chop_value < 45.0
        is_ranging = chop_value > 55.0
        
        # === HTF TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 1H TREND ===
        price_above_hma_21 = close[i] > hma_21[i]
        price_below_hma_21 = close[i] < hma_21[i]
        
        # === CRSI EXTREMES (mean reversion signals) ===
        crsi_oversold = crsi[i] < 20.0  # Looser than 10 for more trades
        crsi_overbought = crsi[i] > 80.0  # Looser than 90 for more trades
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG entries - CRSI mean reversion with HTF trend filter
        long_score = 0
        
        if crsi_oversold:
            long_score += 2  # Strong signal
        if crsi_extreme_oversold:
            long_score += 1  # Extra for extreme
        if rsi_oversold:
            long_score += 1
        if price_above_hma_4h:
            long_score += 1  # HTF trend aligned
        if price_above_hma_1d:
            long_score += 1  # Macro bias positive
        if volume_ok:
            long_score += 0.5
        
        # LONG: Need score >= 3 (looser than previous strategies)
        if long_score >= 3.0:
            if is_trending and price_above_hma_1d:
                # Trending + macro bullish = full size
                new_signal = POSITION_SIZE_FULL
            elif is_ranging:
                # Range regime = mean reversion long
                new_signal = POSITION_SIZE_HALF
            elif price_above_hma_4h:
                # 4h trend positive
                new_signal = POSITION_SIZE_HALF
        
        # SHORT entries - CRSI mean reversion with HTF trend filter
        short_score = 0
        
        if crsi_overbought:
            short_score += 2  # Strong signal
        if crsi_extreme_overbought:
            short_score += 1  # Extra for extreme
        if rsi_overbought:
            short_score += 1
        if price_below_hma_4h:
            short_score += 1  # HTF trend aligned
        if price_below_hma_1d:
            short_score += 1  # Macro bias negative
        if volume_ok:
            short_score += 0.5
        
        # SHORT: Need score >= 3 (looser than previous strategies)
        if short_score >= 3.0:
            if is_trending and price_below_hma_1d:
                # Trending + macro bearish = full size
                new_signal = -POSITION_SIZE_FULL
            elif is_ranging:
                # Range regime = mean reversion short
                new_signal = -POSITION_SIZE_HALF
            elif price_below_hma_4h:
                # 4h trend negative
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid (reduces churn)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 1h HMA
                if price_above_hma_21:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 1h HMA
                if price_below_hma_21:
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
        # Exit long if price crosses below 1h HMA significantly
        if in_position and position_side > 0 and price_below_hma_21:
            new_signal = 0.0
        
        # Exit short if price crosses above 1h HMA significantly
        if in_position and position_side < 0 and price_above_hma_21:
            new_signal = 0.0
        
        # Exit if macro bias flips strongly against position
        if in_position and position_side > 0 and price_below_hma_1d:
            new_signal = 0.0
        if in_position and position_side < 0 and price_above_hma_1d:
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
                # Position flip
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