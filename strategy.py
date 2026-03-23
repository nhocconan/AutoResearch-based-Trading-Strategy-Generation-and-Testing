#!/usr/bin/env python3
"""
Experiment #693: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI + Donchian)

Hypothesis: Daily timeframe with weekly trend bias provides optimal balance between
signal quality and trade frequency. Using Choppiness Index to switch between:
1. MEAN REVERSION regime (CHOP > 61.8): Connors RSI extremes for counter-trend entries
2. TREND FOLLOWING regime (CHOP < 38.2): Donchian breakouts with HTF HMA bias

Why this should work:
- Connors RSI has 75% win rate in academic studies (short-term mean reversion)
- Choppiness Index is proven regime filter (range vs trend detection)
- 1w HMA provides strong trend bias without look-ahead
- 1d TF naturally limits trades to 20-50/year (avoids fee drag)
- Volume filter reduces false breakouts

Key differences from failed experiments:
- Looser CRSI thresholds (25/75 not 10/90) to ensure trade frequency
- Volume confirmation on breakouts (reduces whipsaw)
- Asymmetric sizing: full size in direction of 1w trend, half against
- Simpler exit logic (RSI extremes + time-based)

Target: Sharpe > 0.612, trades >= 40 train, >= 5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_donchian_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long: CRSI < 10-25 | Short: CRSI > 75-90
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(3) on close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.clip(rsi_close, 0, 100)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak (inverted for mean reversion)
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_gain = np.concatenate([[np.nan], avg_streak_gain])
    avg_streak_loss = np.concatenate([[np.nan], avg_streak_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank - where current return ranks vs last 100 days
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns) * 100
            percent_rank[i] = rank
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_close[valid_mask] + rsi_streak[valid_mask] + percent_rank[valid_mask]) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_donchian_channels(high, low, period=20):
    """Donchian Channels - highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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
    
    if n < period:
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

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs rolling median."""
    vol_median = pd.Series(volume).rolling(window=period, min_periods=period).median().values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = volume / (vol_median + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    chop = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    atr_1d = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    FULL_SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    bars_in_trade = 0
    
    for i in range(250, n):  # Need 200 for SMA + buffer for HTF alignment
        # Skip if indicators not ready
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(atr_1d[i]) or np.isnan(donchian_upper[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            continue
        if atr_1d[i] <= 1e-10:
            continue
        
        # === REGIME DETECTION ===
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        # Neutral zone: 38.2 <= CHOP <= 61.8 (reduce position size)
        is_neutral = not is_ranging and not is_trending
        
        # === TREND BIAS (1w HMA) ===
        trend_bullish_1w = close[i] > hma_1w_aligned[i]
        trend_bearish_1w = close[i] < hma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 0.8  # At least 80% of median volume
        
        desired_signal = 0.0
        current_size = FULL_SIZE
        
        # Reduce size in neutral regime
        if is_neutral:
            current_size = HALF_SIZE
        
        # === MEAN REVERSION MODE (Ranging - CHOP > 61.8) ===
        if is_ranging:
            # Long: CRSI oversold + above SMA200 (don't fight major trend)
            if crsi[i] < 25 and close[i] > sma_200[i]:
                # Full size if aligned with 1w trend, half if against
                if trend_bullish_1w:
                    desired_signal = FULL_SIZE
                else:
                    desired_signal = HALF_SIZE
            
            # Short: CRSI overbought + below SMA200
            elif crsi[i] > 75 and close[i] < sma_200[i]:
                if trend_bearish_1w:
                    desired_signal = -FULL_SIZE
                else:
                    desired_signal = -HALF_SIZE
        
        # === TREND FOLLOWING MODE (Trending - CHOP < 38.2) ===
        elif is_trending:
            # Long breakout: price breaks Donchian upper + volume + 1w bullish
            if close[i] >= donchian_upper[i] and volume_confirmed and trend_bullish_1w:
                desired_signal = FULL_SIZE
            
            # Short breakout: price breaks Donchian lower + volume + 1w bearish
            elif close[i] <= donchian_lower[i] and volume_confirmed and trend_bearish_1w:
                desired_signal = -FULL_SIZE
            
            # Weaker signals without volume confirmation
            elif close[i] >= donchian_upper[i] and trend_bullish_1w:
                desired_signal = HALF_SIZE
            elif close[i] <= donchian_lower[i] and trend_bearish_1w:
                desired_signal = -HALF_SIZE
        
        # === NEUTRAL MODE (38.2 <= CHOP <= 61.8) ===
        else:
            # Only take very strong mean reversion signals
            if crsi[i] < 15 and close[i] > sma_200[i] and trend_bullish_1w:
                desired_signal = HALF_SIZE
            elif crsi[i] > 85 and close[i] < sma_200[i] and trend_bearish_1w:
                desired_signal = -HALF_SIZE
        
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
        
        # === EXIT CONDITIONS ===
        # Exit long: CRSI overbought OR trend reverses below 1w HMA
        if in_position and position_side > 0:
            if crsi[i] > 80:
                desired_signal = 0.0
            elif close[i] < hma_1w_aligned[i] and bars_in_trade > 3:
                desired_signal = 0.0
        
        # Exit short: CRSI oversold OR trend reverses above 1w HMA
        if in_position and position_side < 0:
            if crsi[i] < 20:
                desired_signal = 0.0
            elif close[i] > hma_1w_aligned[i] and bars_in_trade > 3:
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions still valid ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if CRSI not overbought and 1w trend intact
                if crsi[i] < 70 and close[i] > hma_1w_aligned[i]:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if CRSI not oversold and 1w trend intact
                if crsi[i] > 30 and close[i] < hma_1w_aligned[i]:
                    desired_signal = -current_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                bars_in_trade = 0
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                bars_in_trade = 0
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
            bars_in_trade += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
                bars_in_trade = 0
        
        signals[i] = desired_signal
    
    return signals