#!/usr/bin/env python3
"""
Experiment #356: 12h Primary + 1d HTF — Connors RSI Mean Reversion with Choppiness Regime

Hypothesis: 12h timeframe with Connors RSI + Choppiness Index worked well in exp #346 (Sharpe=0.151).
This version improves by:
1. Using 1d HMA(21) for cleaner macro bias (proven in exp #351)
2. Connors RSI for mean reversion entries (75% win rate in literature)
3. Choppiness Index to switch between mean revert (CHOP>50) and trend follow (CHOP<50)
4. Donchian breakout for trend regime entries
5. RELAXED thresholds to ensure 25-50 trades/year on 12h
6. ATR(14) trailing stop at 2.5x for risk management

KEY INSIGHT: 12h timeframe naturally filters noise. Connors RSI extremes (<15 or >85) 
are rare enough to avoid overtrading but common enough to generate 30-50 trades/year.
Combined with 1d HMA bias, this should work in both bull and bear markets.

TARGET: 25-50 trades/year on 12h, Sharpe > 0.4 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_1d_hma_donchian_v2"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Fast RSI on price
    RSI(streak, 2): RSI on up/down streak length
    PercentRank(100): Percentile rank of current close vs last 100 closes
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: Fast RSI(3) on price
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI(2) on streak
    # Streak: consecutive up/down bars
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Component 3: PercentRank(100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / len(x.iloc[:-1]) * 100 if len(x) > 1 else 50,
        raw=False
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return highest, lowest

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # Donchian channels for breakout detection
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    # HMA for trend confirmation
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # 30% position size for 12h (target 25-50 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA - HARD FILTER) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 50.0  # High choppiness = range regime (mean revert)
        is_trending = chop[i] <= 50.0  # Low choppiness = trend regime (breakout)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Connors RSI mean reversion
            # Long: CRSI < 15 + price above 1d HMA
            # Short: CRSI > 85 + price below 1d HMA
            
            crsi_oversold = crsi[i] < 15.0
            crsi_overbought = crsi[i] > 85.0
            
            if price_above_hma_1d and crsi_oversold:
                # Long oversold in bullish macro (range regime)
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1d and crsi_overbought:
                # Short overbought in bearish macro (range regime)
                desired_signal = -BASE_SIZE
        
        elif is_trending:
            # TREND REGIME: Donchian breakout + HMA confirmation
            # Long: Price breaks Donchian high + HMA21 > HMA50 + 1d bullish
            # Short: Price breaks Donchian low + HMA21 < HMA50 + 1d bearish
            
            hma_bullish = hma_21[i] > hma_50[i]
            hma_bearish = hma_21[i] < hma_50[i]
            
            # Breakout detection (price at channel edge)
            breakout_long = close[i] >= donchian_high[i] * 0.995  # Near high
            breakout_short = close[i] <= donchian_low[i] * 1.005  # Near low
            
            if price_above_hma_1d and hma_bullish and breakout_long:
                # Long breakout in bullish macro (trend regime)
                desired_signal = BASE_SIZE
            
            elif price_below_hma_1d and hma_bearish and breakout_short:
                # Short breakout in bearish macro (trend regime)
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi[i] > 70:
            # Long position: exit when CRSI reaches overbought
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30:
            # Short position: exit when CRSI reaches oversold
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if regime and bias still valid
            if position_side > 0:
                if price_above_hma_1d:
                    if (is_choppy and crsi[i] < 70) or \
                       (is_trending and hma_21[i] > hma_50[i]):
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                if price_below_hma_1d:
                    if (is_choppy and crsi[i] > 30) or \
                       (is_trending and hma_21[i] < hma_50[i]):
                        desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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