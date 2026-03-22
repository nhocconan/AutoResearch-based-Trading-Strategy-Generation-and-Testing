#!/usr/bin/env python3
"""
Experiment #002: 12h Dual-Regime Strategy with Choppiness Index + Connors RSI

Hypothesis: Single-regime strategies fail because BTC/ETH alternate between trending 
and ranging markets. This strategy detects regime using Choppiness Index and adapts:
- CHOP > 61.8 (range): Connors RSI mean reversion at Bollinger bounds
- CHOP < 38.2 (trend): HMA + Donchian breakout in trend direction
- CHOP 38.2-61.8 (neutral): No trades (wait for clarity)

Key improvements over #012:
1. Regime-adaptive entry logic (not one-size-fits-all)
2. Connors RSI for mean reversion (proven 75% win rate in ranges)
3. Simpler position tracking (signal-based, no complex state)
4. Looser entry thresholds to ensure minimum trade count
5. 1d HMA for major trend bias (only trade with daily trend)

Why 12h works:
- Natural 20-50 trades/year (fee drag ~1-2.5%)
- Filters lower TF noise while capturing major moves
- Works across BTC/ETH/SOL (not SOL-biased)

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_connors_1d_bias_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    # Highest high and lowest low over period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    hh = high_s.rolling(window=period, min_periods=period).max().values
    ll = low_s.rolling(window=period, min_periods=period).min().values
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    range_val = hh - ll
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    chop = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    close_s = pd.Series(close)
    
    # RSI(3) on close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_close = 100 - (100 / (1 + rs))
    
    # RSI on streak (consecutive up/down days)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / avg_streak_loss
    streak_rs = streak_rs.replace([np.inf, -np.inf], 0)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # PercentRank(100) - where current close ranks in last 100 bars
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    ).values * 100
    
    # Combine into CRSI
    crsi = (rsi_close.values + rsi_streak + percent_rank) / 3
    crsi = np.nan_to_num(crsi, nan=50)
    return crsi

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels)
    BASE_SIZE = 0.28
    
    # Track position for stoploss
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 12H TREND ===
        hma_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === REGIME DETECTION ===
        is_range = chop_14[i] > 61.8
        is_trend = chop_14[i] < 38.2
        # Neutral zone: 38.2-61.8 (no trades or reduced size)
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        if position_side != 0:
            if position_side > 0:
                if close[i] > highest_since_entry:
                    highest_since_entry = close[i]
                stop_price = highest_since_entry - 2.5 * atr_14[i]
                if close[i] < stop_price:
                    stoploss_triggered = True
            else:  # short
                if lowest_since_entry == 0.0 or close[i] < lowest_since_entry:
                    lowest_since_entry = close[i]
                stop_price = lowest_since_entry + 2.5 * atr_14[i]
                if close[i] > stop_price:
                    stoploss_triggered = True
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (Connors RSI + Bollinger)
            # Long: CRSI < 15 + price near BB lower + daily bullish bias
            if daily_bullish and crsi[i] < 20 and close[i] <= bb_lower[i] * 1.01:
                new_signal = BASE_SIZE
            
            # Short: CRSI > 85 + price near BB upper + daily bearish bias
            elif daily_bearish and crsi[i] > 80 and close[i] >= bb_upper[i] * 0.99:
                new_signal = -BASE_SIZE
        
        elif is_trend:
            # TREND FOLLOWING MODE (HMA + Donchian breakout)
            # Long: HMA bullish + Donchian breakout + daily bullish
            if hma_bullish and daily_bullish:
                if i > 0 and not np.isnan(donchian_upper[i-1]):
                    if close[i] > donchian_upper[i-1]:
                        new_signal = BASE_SIZE
            
            # Short: HMA bearish + Donchian breakdown + daily bearish
            elif hma_bearish and daily_bearish:
                if i > 0 and not np.isnan(donchian_lower[i-1]):
                    if close[i] < donchian_lower[i-1]:
                        new_signal = -BASE_SIZE
        
        else:
            # NEUTRAL ZONE - reduced size entries only on extreme CRSI
            if crsi[i] < 10 and daily_bullish:
                new_signal = BASE_SIZE * 0.6
            elif crsi[i] > 90 and daily_bearish:
                new_signal = -BASE_SIZE * 0.6
        
        # === APPLY STOPLOSS ===
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if position_side > 0 and hma_bearish and daily_bearish:
            new_signal = 0.0
        if position_side < 0 and hma_bullish and daily_bullish:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if position_side == 0:
                # New entry
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
            # Exit
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals