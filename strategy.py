#!/usr/bin/env python3
"""
Experiment #186: 12h Primary + 1d HTF — Simplified Regime + CRSI + Donchian

Hypothesis: Previous 12h strategies failed due to (1) overly complex state machines
causing logic bugs, (2) CRSI thresholds too extreme (<10, >90) resulting in 0 trades,
(3) hold logic conflicting with entry logic causing signal churn.

This strategy SIMPLIFIES the approach:
1. Clear regime detection via Choppiness Index (no overlap zones)
2. Looser CRSI thresholds (15/85 instead of 10/90) for more trades
3. No complex position tracking - signal directly reflects desired state
4. 1d HMA as simple trend bias filter
5. ATR stoploss via signal reset (clean implementation)
6. Volume confirmation on breakouts to reduce false signals

Key improvements over #184:
- Simpler signal logic (no state machine bugs)
- CRSI thresholds relaxed for 30-50 trades/year target
- Volume filter on Donchian breakouts (breakout + volume > 1.5x avg)
- Cleaner stoploss implementation

TARGET: 25-45 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_crsi_donchian_1d_v2"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Using 55/45 thresholds for clearer regime separation.
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 15-20, Short when CRSI > 80-85
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100.0 - (100.0 / (1.0 + rs))
    rsi_short = rsi_short.fillna(50.0).values
    
    # RSI of Streak (consecutive up/down bars)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 100.0 * min(abs(streak[i]), streak_period) / streak_period
        elif streak[i] < 0:
            streak_rsi[i] = 100.0 * (1.0 - min(abs(streak[i]), streak_period) / streak_period)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns[:-1] < current_return)
            percent_rank[i] = 100.0 * rank / max(len(returns) - 1, 1)
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bands)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_volume_sma(volume, period=20):
    """Calculate simple moving average of volume."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    
    # Position sizing (discrete levels to minimize fee churn)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # Track for stoploss
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] < 1e-10:
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] < 1e-10:
            continue
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (clear separation, no overlap) ===
        is_range = chop_14[i] >= 55.0
        is_trend = chop_14[i] <= 45.0
        # Neutral zone: 45 < CHOP < 55 (no new entries, hold existing)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (Connors RSI)
            # Long: CRSI < 20 + price above 1d HMA
            if crsi[i] < 20.0 and price_above_hma_1d:
                new_signal = SIZE_FULL
            
            # Short: CRSI > 80 + price below 1d HMA
            elif crsi[i] > 80.0 and price_below_hma_1d:
                new_signal = -SIZE_FULL
        
        elif is_trend:
            # TREND FOLLOWING MODE (Donchian Breakout + Volume)
            volume_confirmed = volume[i] > 1.3 * vol_sma_20[i]
            
            # Long: Price breaks Donchian upper + volume + price above 1d HMA
            if close[i] > donchian_upper[i-1] and volume_confirmed and price_above_hma_1d:
                new_signal = SIZE_FULL
            
            # Short: Price breaks Donchian lower + volume + price below 1d HMA
            elif close[i] < donchian_lower[i-1] and volume_confirmed and price_below_hma_1d:
                new_signal = -SIZE_FULL
        
        # === HOLD LOGIC (simplified) ===
        # If we have a position from previous bar, hold it unless exit conditions met
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if prev_signal != 0.0 and new_signal == 0.0:
            # Check if we should hold the position
            if prev_signal > 0:
                # Hold long if price still above 1d HMA and not in opposite regime
                if price_above_hma_1d and not (is_range and crsi[i] > 50):
                    new_signal = prev_signal
            elif prev_signal < 0:
                # Hold short if price still below 1d HMA and not in opposite regime
                if price_below_hma_1d and not (is_range and crsi[i] < 50):
                    new_signal = prev_signal
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if prev_signal > 0:
            # Update highest since entry for long positions
            if entry_price[i-1] > 0:
                highest_since_entry[i] = max(highest_since_entry[i-1], close[i])
            else:
                highest_since_entry[i] = close[i]
                entry_price[i] = close[i]
            
            stop_price = highest_since_entry[i] - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        elif prev_signal < 0:
            # Update lowest since entry for short positions
            if entry_price[i-1] < 0:
                if lowest_since_entry[i-1] == 0:
                    lowest_since_entry[i] = close[i]
                else:
                    lowest_since_entry[i] = min(lowest_since_entry[i-1], close[i])
            else:
                lowest_since_entry[i] = close[i]
                entry_price[i] = close[i]
            
            if lowest_since_entry[i] > 0:
                stop_price = lowest_since_entry[i] + 2.5 * atr_14[i]
                if close[i] > stop_price:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
            entry_price[i] = 0.0
            highest_since_entry[i] = 0.0
            lowest_since_entry[i] = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below 1d HMA
        if prev_signal > 0 and price_below_hma_1d:
            new_signal = 0.0
            entry_price[i] = 0.0
        
        # Exit short if price crosses above 1d HMA
        if prev_signal < 0 and price_above_hma_1d:
            new_signal = 0.0
            entry_price[i] = 0.0
        
        # === UPDATE ENTRY PRICE TRACKING ===
        if new_signal != 0.0 and entry_price[i] == 0.0:
            entry_price[i] = close[i]
            if new_signal > 0:
                highest_since_entry[i] = close[i]
            else:
                lowest_since_entry[i] = close[i]
        elif new_signal == 0.0:
            entry_price[i] = 0.0
            highest_since_entry[i] = 0.0
            lowest_since_entry[i] = 0.0
        else:
            # Carry forward tracking values
            if i > 0:
                if entry_price[i] == 0.0:
                    entry_price[i] = entry_price[i-1]
                if highest_since_entry[i] == 0.0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                if lowest_since_entry[i] == 0.0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        
        signals[i] = new_signal
    
    return signals