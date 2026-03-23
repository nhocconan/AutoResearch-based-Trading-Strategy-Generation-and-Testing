#!/usr/bin/env python3
"""
Experiment #352: 12h Primary + 1d/1w HTF — Dual Regime with Connors RSI + Donchian

Hypothesis: 12h timeframe is underutilized but proven to work (Exp #346 Sharpe=0.151).
Previous 12h strategies failed due to:
1. Too strict entry filters (0 trades = auto-reject)
2. Missing proper regime adaptation (trend vs range needs different logic)
3. No weekly macro bias filter for stronger directional conviction

This strategy combines proven patterns:
1. 1d HMA(21) + 1w HMA(21) for MACRO BIAS (dual HTF confirmation)
2. 12h Choppiness Index for regime (CHOP>55=range, CHOP<45=trend, 45-55=neutral)
3. RANGE REGIME: Connors RSI <15 (long) or >85 (short) + macro bias alignment
4. TREND REGIME: Donchian(20) breakout + HMA(16/48) confirmation + macro bias
5. ATR(14) trailing stop at 2.5x for risk management
6. RELAXED thresholds to ensure 20-50 trades/year on 12h

KEY INSIGHT: Connors RSI (CRSI) has 75% win rate for mean reversion in range markets.
Donchian breakouts work in trending markets. Choppiness Index tells us which regime.
Dual HTF (1d + 1w HMA) provides stronger bias than single HTF.

TARGET: 20-50 trades/year on 12h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_donchian_1d1w_hma_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Entry: CRSI < 10 (long), CRSI > 90 (short)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(close, 3)
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # RSI(streak, 2) - streak is consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.maximum(streak, 0)
    streak_loss = np.abs(np.minimum(streak, 0))
    
    # Simple streak RSI calculation
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        avg_gain = np.mean(streak_gain[max(0, i-streak_period+1):i+1])
        avg_loss = np.mean(streak_loss[max(0, i-streak_period+1):i+1])
        if avg_loss == 0:
            streak_rsi[i] = 100.0
        else:
            rs = avg_gain / (avg_loss + 1e-10)
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # PercentRank(100) - where current close ranks in last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period+1):i+1]
        rank = np.sum(window[:-1] < window[-1])
        percent_rank[i] = (rank / (rank_period - 1)) * 100.0
    
    # Combine into CRSI
    crsi = (rsi_close + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range, CHOP < 38.2 = trend
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # HMA for trend detection (fast and slow)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    
    # Donchian channels for breakout detection
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for stronger macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 12h (target 20-50 trades/year)
    
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
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (Dual HTF: 1d + 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # Strong bias: both 1d and 1w agree
        strong_bullish = price_above_hma_1d and price_above_hma_1w
        strong_bearish = price_below_hma_1d and price_below_hma_1w
        # Weak bias: only 1d agrees (1w neutral or opposite)
        weak_bullish = price_above_hma_1d and not price_above_hma_1w
        weak_bearish = price_below_hma_1d and not price_below_hma_1w
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # High choppiness = range regime (mean revert)
        is_trending = chop[i] < 45.0  # Low choppiness = trend regime (breakout)
        # 45-55 = neutral, use trend logic as default
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Connors RSI mean reversion
            # Long: CRSI < 15 + strong/weak bullish bias
            # Short: CRSI > 85 + strong/weak bearish bias
            
            crsi_oversold = crsi[i] < 15.0
            crsi_overbought = crsi[i] > 85.0
            
            if crsi_oversold and (strong_bullish or weak_bullish):
                # Long oversold in bullish macro (range regime)
                desired_signal = BASE_SIZE
            
            elif crsi_overbought and (strong_bearish or weak_bearish):
                # Short overbought in bearish macro (range regime)
                desired_signal = -BASE_SIZE
        
        elif is_trending:
            # TREND REGIME: Donchian breakout + HMA confirmation
            # Long: Price breaks Donchian high + HMA16 > HMA48 + bullish bias
            # Short: Price breaks Donchian low + HMA16 < HMA48 + bearish bias
            
            breakout_long = close[i] > donchian_high[i-1]  # Break above previous high
            breakout_short = close[i] < donchian_low[i-1]  # Break below previous low
            
            hma_bullish = hma_16[i] > hma_48[i]
            hma_bearish = hma_16[i] < hma_48[i]
            
            if breakout_long and hma_bullish and (strong_bullish or weak_bullish):
                # Long breakout in bullish macro (trend regime)
                desired_signal = BASE_SIZE
            
            elif breakout_short and hma_bearish and (strong_bearish or weak_bearish):
                # Short breakout in bearish macro (trend regime)
                desired_signal = -BASE_SIZE
        
        else:
            # NEUTRAL REGIME (45-55): Use trend logic as default but require stronger confirmation
            hma_bullish = hma_16[i] > hma_48[i]
            hma_bearish = hma_16[i] < hma_48[i]
            
            if hma_bullish and strong_bullish:
                desired_signal = BASE_SIZE * 0.7
            elif hma_bearish and strong_bearish:
                desired_signal = -BASE_SIZE * 0.7
        
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
        
        # === CRSI EXIT (mean reversion complete in range regime) ===
        if is_choppy and in_position:
            if position_side > 0 and crsi[i] > 70:
                # Long position: exit when CRSI reaches overbought
                desired_signal = 0.0
            
            if position_side < 0 and crsi[i] < 30:
                # Short position: exit when CRSI reaches oversold
                desired_signal = 0.0
        
        # === HMA EXIT (trend reversal in trend regime) ===
        if is_trending and in_position:
            if position_side > 0 and hma_16[i] < hma_48[i]:
                # Long position: exit when HMA turns bearish
                desired_signal = 0.0
            
            if position_side < 0 and hma_16[i] > hma_48[i]:
                # Short position: exit when HMA turns bullish
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            # Check if regime and bias still valid
            if position_side > 0:
                if strong_bullish or weak_bullish:
                    if (is_choppy and crsi[i] < 70) or \
                       (is_trending and hma_16[i] > hma_48[i]):
                        desired_signal = BASE_SIZE
            elif position_side < 0:
                if strong_bearish or weak_bearish:
                    if (is_choppy and crsi[i] > 30) or \
                       (is_trending and hma_16[i] < hma_48[i]):
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