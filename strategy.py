#!/usr/bin/env python3
"""
Experiment #029: 4h Primary + 1d HTF — Connors RSI + KAMA Adaptive Trend + Choppiness Regime

Hypothesis: 4h timeframe with daily trend bias will capture medium-term moves while avoiding
the whipsaw of lower timeframes. Connors RSI provides proven mean-reversion edge (75% win rate),
while KAMA adapts to volatility changes better than EMA/HMA. Choppiness Index switches between
mean-reversion (range) and trend-following regimes.

Key improvements over failed experiments:
1. LOOSER entry conditions (CRSI < 20 or > 80, not extreme 10/90) to guarantee trades
2. KAMA instead of HMA — more adaptive to volatility regimes
3. Volume confirmation on breakouts (reduces false signals)
4. 1d KAMA for macro trend bias (stronger than 1w for 4h trading)
5. Discrete position sizing (0.25) to minimize fee churn

Why 4h works: 20-50 trades/year target, less fee drag than 1h/30m, more signals than 1d/12h.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_kama_chop_regime_1d_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Price change over period
    price_change = np.abs(close - np.roll(close, period))
    price_change[:period] = np.nan
    
    # Sum of absolute price changes (volatility)
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1))[1:])
    
    # Efficiency Ratio (ER)
    er = price_change / (volatility + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Dynamic smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Proven mean-reversion indicator with 75% win rate on crypto.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value
    for i in range(streak_period, n):
        streak_window = streak[max(0, i-streak_period+1):i+1]
        up_streaks = np.sum(streak_window > 0)
        streak_rsi[i] = (up_streaks / (streak_period + 1e-10)) * 100.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns = np.diff(close[max(0, i-rank_period):i+1])
        if len(returns) > 0:
            current_return = close[i] - close[i-1]
            percent_rank[i] = (np.sum(returns < current_return) / len(returns)) * 100.0
    
    # Combine into CRSI
    crsi = (rsi_3.values + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for macro trend bias
    kama_1d = calculate_kama(df_1d['close'].values, period=21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    kama_21 = calculate_kama(close, period=21)
    kama_50 = calculate_kama(close, period=50)
    
    # Volume moving average for confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(120, n):  # Need enough data for all indicators
        # Skip if indicators not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(kama_21[i]) or np.isnan(kama_50[i]) or atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market
        is_trending = chop_value < 45.0  # Trend market (with hysteresis)
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 25.0  # Mean reversion long
        crsi_overbought = crsi[i] > 75.0  # Mean reversion short
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === KAMA TREND ===
        kama_bullish = kama_21[i] > kama_50[i]
        kama_bearish = kama_21[i] < kama_50[i]
        kama_slope_up = kama_21[i] > kama_21[i-10] if i > 10 else False
        kama_slope_down = kama_21[i] < kama_21[i-10] if i > 10 else False
        
        # === VOLUME CONFIRMATION ===
        volume_above_avg = volume[i] > vol_sma_20[i] * 1.2 if not np.isnan(vol_sma_20[i]) else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion (CRSI + BB) ---
        if is_ranging:
            # Long: CRSI oversold + price below BB lower + daily bias helps
            if crsi_oversold or price_below_bb_lower:
                if price_above_kama_1d or crsi_rising:  # Daily bullish OR CRSI turning up
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + price above BB upper + daily bias helps
            elif crsi_overbought or price_above_bb_upper:
                if price_below_kama_1d or crsi_falling:  # Daily bearish OR CRSI turning down
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following (KAMA + Volume) ---
        elif is_trending:
            # Long: KAMA bullish + price above KAMA21 + volume confirmation
            if kama_bullish and close[i] > kama_21[i]:
                if price_above_kama_1d and kama_slope_up:  # Daily + 4h trend aligned
                    if volume_above_avg or crsi_rising:  # Volume or momentum confirmation
                        new_signal = POSITION_SIZE
            
            # Short: KAMA bearish + price below KAMA21 + volume confirmation
            elif kama_bearish and close[i] < kama_21[i]:
                if price_below_kama_1d and kama_slope_down:  # Daily + 4h trend aligned
                    if volume_above_avg or crsi_falling:  # Volume or momentum confirmation
                        new_signal = -POSITION_SIZE
        
        # --- FALLBACK: CRSI extreme reversal (always check) ---
        if new_signal == 0.0:
            # Very oversold CRSI = long regardless of regime
            if crsi[i] < 15.0 and crsi_rising:
                new_signal = POSITION_SIZE
            # Very overbought CRSI = short regardless of regime
            elif crsi[i] > 85.0 and crsi_falling:
                new_signal = -POSITION_SIZE
        
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
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if daily trend turns bearish
        if in_position and position_side > 0:
            if price_below_kama_1d and kama_bearish:
                new_signal = 0.0
        
        # Exit short if daily trend turns bullish
        if in_position and position_side < 0:
            if price_above_kama_1d and kama_bullish:
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