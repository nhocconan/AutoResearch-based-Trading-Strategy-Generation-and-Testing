#!/usr/bin/env python3
"""
Experiment #247: 1d Primary + 4h HTF — Simplified Dual Regime with Looser Entries

Hypothesis: Previous 1d strategies (#237, #243, #246) failed due to:
1. Too many confluence filters = 0 or very few trades
2. Extreme thresholds (CHOP > 61.8, CRSI < 15) rarely trigger
3. 1w HTF too slow for 1d primary

This version simplifies:
- HTF: 4h instead of 1w (faster trend confirmation)
- CHOP thresholds: 55/45 instead of 61.8/38.2 (more regime switches)
- CRSI thresholds: 25/75 instead of 15/85 (more mean reversion signals)
- Donchian(10) instead of Donchian(20) (more breakouts)
- Remove neutral regime (only range/trend binary)
- ATR stop: 2.0x instead of 2.5x (tighter risk control)

TARGET: 30-50 trades/year on 1d, Sharpe > 0.5 on ALL symbols
CRITICAL: Entry conditions MUST be loose enough to generate >30 trades in train period
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_simplified_regime_crsi_donchian_4h_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # PercentRank(100)
    returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current_return = returns.iloc[i]
        if pd.isna(current_return):
            percent_rank[i] = 50.0
        else:
            rank = (window < current_return).sum()
            percent_rank[i] = 100.0 * rank / rank_period
    
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 55 = range/choppy market
    CHOP < 45 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop[:period] = 50.0
    return chop

def calculate_donchian(high, low, period=10):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    upper[:period] = np.nan
    lower[:period] = np.nan
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1d indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=10)
    
    # Calculate 4h HMA for trend confirmation (aligned properly)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_21[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htfbullish = close[i] > hma_4h_aligned[i]
        htfbearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (Simplified Choppiness) ===
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        # === PRIMARY TREND (1d HMA) ===
        hma_bullish = close[i] > hma_21[i]
        hma_bearish = close[i] < hma_21[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # RANGE REGIME: Connors RSI mean reversion (LOOSER thresholds)
        if is_range:
            # Long: CRSI < 25 (oversold)
            if crsi[i] < 25.0:
                if htfbullish or hma_bullish:
                    desired_signal = POSITION_SIZE
                else:
                    desired_signal = POSITION_SIZE * 0.67  # smaller position against HTF
            # Short: CRSI > 75 (overbought)
            elif crsi[i] > 75.0:
                if htfbearish or hma_bearish:
                    desired_signal = -POSITION_SIZE
                else:
                    desired_signal = -POSITION_SIZE * 0.67
        
        # TREND REGIME: Donchian breakout + HMA confirmation
        elif is_trend:
            # Long breakout
            if close[i] >= donchian_upper[i]:
                if hma_bullish:
                    desired_signal = POSITION_SIZE
                elif htfbullish:
                    desired_signal = POSITION_SIZE * 0.67
            # Short breakout
            elif close[i] <= donchian_lower[i]:
                if hma_bearish:
                    desired_signal = -POSITION_SIZE
                elif htfbearish:
                    desired_signal = -POSITION_SIZE * 0.67
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_bearish and is_trend:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish and is_trend:
            desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 65.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 35.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if (is_range and crsi[i] < 65.0) or (is_trend and hma_bullish):
                    desired_signal = POSITION_SIZE * 0.67
            elif position_side < 0:
                if (is_range and crsi[i] > 35.0) or (is_trend and hma_bearish):
                    desired_signal = -POSITION_SIZE * 0.67
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals