#!/usr/bin/env python3
"""
Experiment #197: 1d Primary + 1w HTF — Choppiness Regime + Connors RSI + Donchian Breakout

Hypothesis: Daily timeframe with weekly macro bias can capture major regime shifts while
avoiding noise. Combining Choppiness Index for regime detection with Connors RSI for
mean reversion and Donchian breakouts for trend following should work across bull/bear/range.

Key components:
1. Choppiness Index (14): CHOP > 55 = range (mean revert), CHOP < 45 = trend (breakout)
2. Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - proven 75% win rate
3. Donchian(20) breakout: captures sustained moves in trend regime
4. 1w HMA(21): macro bias - only trade with weekly trend
5. ATR(14) trailing stop: 2.5x ATR for risk management

Asymmetric logic:
- With HTF trend: CRSI(15/85) + Donchian breakout
- Against HTF trend: CRSI(8/92) + volume confirmation (harder to enter counter-trend)

TARGET: 25-45 trades/year, Sharpe > 0.5 on ALL symbols
Position sizing: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_donchian_1w_v1"
timeframe = "1d"
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
    Proven mean reversion indicator with 75% win rate.
    Long: CRSI < 10-15, Short: CRSI > 85-90
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.zeros(n)
    for i in range(1, n):
        streak = 0
        if close[i] > close[i-1]:
            # Count consecutive up days
            j = i
            while j > 0 and close[j] > close[j-1]:
                streak += 1
                j -= 1
        elif close[i] < close[i-1]:
            # Count consecutive down days (negative)
            j = i
            while j > 0 and close[j] < close[j-1]:
                streak -= 1
                j -= 1
        # Convert streak to RSI-like value (0-100)
        if streak > 0:
            streak_rsi[i] = min(100.0, streak * 50.0)
        elif streak < 0:
            streak_rsi[i] = max(0.0, 100.0 + streak * 50.0)
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            rank = np.sum(returns < current_return) / len(returns)
            percent_rank[i] = rank * 100.0
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    for i in range(max(rsi_period, streak_period, rank_period), n):
        crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Using 55/45 thresholds for better trade frequency.
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            tr_sum += tr
        
        range_val = highest - lowest
        if range_val < 1e-10 or tr_sum < 1e-10:
            chop[i] = 50.0
        else:
            chop[i] = 100.0 * np.log10(tr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume MA for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w HMA for macro trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
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
        if np.isnan(chop_14[i]):
            continue
        if np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ma20[i]):
            continue
        
        # === HTF MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        is_range = chop_14[i] > 55.0
        is_trend = chop_14[i] < 45.0
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma20[i] if vol_ma20[i] > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if is_range:
            # MEAN REVERSION MODE (Connors RSI extremes)
            # Long: CRSI < 15 + price above weekly HMA (with trend) or CRSI < 10 (any)
            if crsi[i] < 15:
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE_FULL
                elif crsi[i] < 10:
                    new_signal = POSITION_SIZE_HALF
            
            # Short: CRSI > 85 + price below weekly HMA (with trend) or CRSI > 90 (any)
            elif crsi[i] > 85:
                if price_below_hma_1w:
                    new_signal = -POSITION_SIZE_FULL
                elif crsi[i] > 90:
                    new_signal = -POSITION_SIZE_HALF
        
        elif is_trend:
            # TREND FOLLOWING MODE (Donchian breakout)
            # Long: Price breaks Donchian upper + weekly HMA bullish
            if close[i] >= donchian_upper[i-1] and price_above_hma_1w:
                new_signal = POSITION_SIZE_FULL
            elif close[i] >= donchian_upper[i-1] and volume_spike:
                new_signal = POSITION_SIZE_HALF
            
            # Short: Price breaks Donchian lower + weekly HMA bearish
            elif close[i] <= donchian_lower[i-1] and price_below_hma_1w:
                new_signal = -POSITION_SIZE_FULL
            elif close[i] <= donchian_lower[i-1] and volume_spike:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not overbought and price above weekly HMA
                if crsi[i] < 70 and price_above_hma_1w:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not oversold and price below weekly HMA
                if crsi[i] > 30 and price_below_hma_1w:
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
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        if in_position and position_side > 0:
            # Long: exit if trend regime starts and price below weekly HMA
            if is_trend and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Short: exit if trend regime starts and price above weekly HMA
            if is_trend and price_above_hma_1w:
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