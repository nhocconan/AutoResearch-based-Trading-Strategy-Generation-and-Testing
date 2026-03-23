#!/usr/bin/env python3
"""
Experiment #182: 12h Primary + 1d/1w HTF — Regime-Adaptive (Choppiness + Connors RSI + HMA)

Hypothesis: Previous 12h strategies failed due to either (1) too strict filters = 0 trades,
or (2) single-regime logic that doesn't adapt to market conditions. This strategy uses
Choppiness Index to detect regime, then applies different logic:
- CHOP > 61.8 (range): Connors RSI mean reversion at extremes
- CHOP < 38.2 (trend): HMA crossover + RSI pullback entries
- 38.2-61.8 (neutral): reduced position size, wait for confirmation

KEY IMPROVEMENTS:
1. Choppiness Index (14-period) for regime detection - proven meta-filter for bear markets
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - 75% win rate
3. 1d HMA for macro bias (only long when price > 1d HMA)
4. 1w HMA for ultra-long-term trend filter
5. ATR trailing stop at 2.5x for risk management
6. Discrete position sizing: 0.0, ±0.20, ±0.30 to minimize fee churn
7. Regime-adaptive: different entry logic per market condition

TARGET: 25-45 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_crsi_chop_hma_1d1w_v1"
timeframe = "12h"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range-bound market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            chop[i] = 100.0
            continue
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    rsi_3 = rsi_3.fillna(50.0).values
    
    # Streak RSI component
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = streak_rsi.fillna(50.0).values
    
    # Percent Rank component
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        lookback = close[i-pr_period+1:i+1]
        current = close[i]
        rank = np.sum(lookback[:-1] < current)
        percent_rank[i] = 100.0 * rank / (pr_period - 1)
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness_index(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    
    # HMA for trend detection
    hma_21 = calculate_hma(close, period=21)
    hma_48 = calculate_hma(close, period=48)
    
    # Volume average (20-bar)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1w HMA for ultra-long-term trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    POSITION_SIZE_QUARTER = 0.10
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop[i] > 61.8  # Range-bound market
        regime_trend = chop[i] < 38.2  # Trending market
        regime_neutral = not regime_range and not regime_trend  # 38.2-61.8
        
        # === HMA TREND (12h) ===
        hma_bullish = hma_21[i] > hma_48[i]
        hma_bearish = hma_21[i] < hma_48[i]
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.5 * vol_avg[i]
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        new_signal = 0.0
        position_size = POSITION_SIZE_HALF  # Default
        
        # LONG entries
        long_signal = False
        
        if regime_range:
            # Mean reversion in range: CRSI < 15 (oversold)
            crsi_oversold = crsi[i] < 15.0
            near_support = close[i] < donchian_lower[i] * 1.02  # Near lower band
            if crsi_oversold and near_support and price_above_hma_1d and volume_ok:
                long_signal = True
                position_size = POSITION_SIZE_HALF
        elif regime_trend:
            # Trend following: HMA bullish + RSI pullback
            rsi_pullback = (rsi_14[i] > 40.0) and (rsi_14[i] < 60.0)
            hma_crossover = hma_bullish
            if hma_crossover and rsi_pullback and price_above_hma_1d and volume_ok:
                long_signal = True
                if price_above_hma_1w:
                    position_size = POSITION_SIZE_FULL
                else:
                    position_size = POSITION_SIZE_HALF
        else:
            # Neutral regime: wait for strong confirmation
            crsi_very_oversold = crsi[i] < 10.0
            hma_strong_bullish = hma_bullish and (hma_21[i] > hma_48[i] * 1.005)
            if (crsi_very_oversold or hma_strong_bullish) and price_above_hma_1d and volume_ok:
                long_signal = True
                position_size = POSITION_SIZE_QUARTER
        
        # SHORT entries
        short_signal = False
        
        if regime_range:
            # Mean reversion in range: CRSI > 85 (overbought)
            crsi_overbought = crsi[i] > 85.0
            near_resistance = close[i] > donchian_upper[i] * 0.98  # Near upper band
            if crsi_overbought and near_resistance and price_below_hma_1d and volume_ok:
                short_signal = True
                position_size = POSITION_SIZE_HALF
        elif regime_trend:
            # Trend following: HMA bearish + RSI pullback
            rsi_pullback = (rsi_14[i] > 40.0) and (rsi_14[i] < 60.0)
            hma_crossover = hma_bearish
            if hma_crossover and rsi_pullback and price_below_hma_1d and volume_ok:
                short_signal = True
                if price_below_hma_1w:
                    position_size = POSITION_SIZE_FULL
                else:
                    position_size = POSITION_SIZE_HALF
        else:
            # Neutral regime: wait for strong confirmation
            crsi_very_overbought = crsi[i] > 90.0
            hma_strong_bearish = hma_bearish and (hma_21[i] < hma_48[i] * 0.995)
            if (crsi_very_overbought or hma_strong_bearish) and price_below_hma_1d and volume_ok:
                short_signal = True
                position_size = POSITION_SIZE_QUARTER
        
        if long_signal:
            new_signal = position_size
        elif short_signal:
            new_signal = -position_size
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above 1d HMA
                if price_above_hma_1d:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below 1d HMA
                if price_below_hma_1d:
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
        # Exit long if price crosses below 1d HMA (trend changed)
        if in_position and position_side > 0 and price_below_hma_1d:
            new_signal = 0.0
        
        # Exit short if price crosses above 1d HMA (trend changed)
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