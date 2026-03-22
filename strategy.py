#!/usr/bin/env python3
"""
Experiment #203: 12h Connors RSI + Triple Timeframe HMA + Choppiness Regime + ATR Stop

Hypothesis: 12h timeframe with Connors RSI (CRSI) captures short-term extremes within
multi-timeframe trend context. CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
provides better entry timing than standard RSI. Choppiness Index filters regime:
CHOP > 61.8 = range (use mean reversion), CHOP < 38.2 = trend (use breakout).
Triple HMA alignment (12h close vs 1d HMA vs 1w HMA) ensures we trade with macro trend.

Why this might work where others failed:
- #191 (12h KAMA): Sharpe=-0.443 - KAMA too slow for 12h entries
- #197 (12h Donchian): Sharpe=-0.192 - breakouts fail in choppy markets
- CRSI catches oversold/overbought within trend (75% win rate in literature)
- Choppiness filter avoids trading breakouts in ranges (major failure mode)
- 1w HMA adds ultra-HTF bias missing from previous 12h attempts
- More flexible entries (CRSI < 30 or > 70, not extreme 10/90) ensure trade count

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels with volatility scaling
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_1d_1w_hma_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(streak): RSI of consecutive up/down days
    PercentRank: percentage of past returns lower than current return
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100 - (100 / (1 + rs))
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    
    # Component 3: PercentRank(100) - percentage of past returns lower than current
    returns = close_s.pct_change()
    percent_rank = pd.Series(index=returns.index, dtype=float)
    
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current_return = returns.iloc[i]
        if pd.notna(current_return):
            rank = (window < current_return).sum() / rank_period * 100
            percent_rank.iloc[i] = rank
        else:
            percent_rank.iloc[i] = 50.0
    
    # Fill initial values
    percent_rank.iloc[:rank_period] = 50.0
    
    # Combine components
    crsi = (rsi_close.values + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    # Calculate ATR for each bar
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
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    # Fill initial values
    choppiness[:period] = 50.0
    
    return choppiness

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness = calculate_choppiness(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        
        # === TRIPLE TIMEFRAME TREND BIAS ===
        # 1w HMA = ultra long-term trend (macro bias)
        # 1d HMA = medium-term trend
        # Price vs both = alignment check
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both HTF agree
        strong_bull = bull_trend_1w and bull_trend_1d
        strong_bear = bear_trend_1w and bear_trend_1d
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = range/choppy (favor mean reversion)
        # CHOP < 38.2 = trending (favor momentum)
        # CHOP 38.2-61.8 = transition (reduce position or stay flat)
        is_choppy = choppiness[i] > 55.0  # Slightly lower threshold for more trades
        is_trending = choppiness[i] < 45.0  # Slightly higher threshold for more trades
        
        # === CONNORS RSI ENTRY SIGNALS ===
        # CRSI < 30 = oversold (long opportunity in uptrend)
        # CRSI > 70 = overbought (short opportunity in downtrend)
        # More flexible than extreme 10/90 to ensure trade count
        crsi_oversold = crsi[i] < 35.0
        crsi_overbought = crsi[i] > 65.0
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG: Strong bull trend OR (trending regime + any bull bias) + CRSI oversold
        if crsi_oversold:
            if strong_bull:
                # Strong trend + oversold = high probability long
                new_signal = SIZE_BASE
            elif is_trending and bull_trend_1d:
                # Trending regime + 1d bull + oversold = good long
                new_signal = SIZE_BASE
            elif is_choppy and bull_trend_1w:
                # Choppy regime + 1w bull + oversold = mean reversion long
                new_signal = SIZE_BASE * 0.8  # Smaller size in chop
        
        # SHORT: Strong bear trend OR (trending regime + any bear bias) + CRSI overbought
        if crsi_overbought:
            if strong_bear:
                # Strong trend + overbought = high probability short
                new_signal = -SIZE_BASE
            elif is_trending and bear_trend_1d:
                # Trending regime + 1d bear + overbought = good short
                new_signal = -SIZE_BASE
            elif is_choppy and bear_trend_1w:
                # Choppy regime + 1w bear + overbought = mean reversion short
                new_signal = -SIZE_BASE * 0.8  # Smaller size in chop
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === EXIT ON REGIME CHANGE ===
        # If in long position and regime becomes strongly choppy with bearish HTF, exit
        if in_position and position_side > 0:
            if choppiness[i] > 65.0 and bear_trend_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if choppiness[i] > 65.0 and bull_trend_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction, update extremes
            elif position_side > 0 and close[i] > highest_close:
                highest_close = close[i]
            elif position_side < 0 and (lowest_close == 0.0 or close[i] < lowest_close):
                lowest_close = close[i]
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals