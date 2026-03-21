#!/usr/bin/env python3
"""
Experiment #183: 1h Connors RSI Mean Reversion with 4h/12h HMA Trend Filter
Hypothesis: Connors RSI (CRSI) captures short-term oversold/overbought conditions
with 75%+ win rate in backtests. Combined with 4h HMA for trend bias and 12h
Choppiness Index for regime detection. Long when CRSI<15 + price>4h_HMA (bullish bias),
short when CRSI>85 + price<4h_HMA (bearish bias). This works in both bull (2021-2024)
and bear/range (2025) markets by only taking mean-reversion trades in trend direction.
Position sizing: 0.30 entry, stoploss at 2*ATR. Discrete levels minimize fee churn.
Timeframe: 1h (required for this experiment) with 4h/12h HTF filters.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_12h_hma_chop_mean_reversion_v1"
timeframe = "1h"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    Reference: Connors et al. "Short Term Trading Strategies That Work"
    Entry: CRSI < 10 (oversold) or CRSI > 90 (overbought)
    """
    n = len(close)
    
    # RSI(3) - very fast RSI
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percentile Rank - where does current close rank vs last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl > 0, range_hl, 1e-10)
    
    # Sum ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    chop = 100 * np.log10(atr_sum / (range_hl * period))
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2
        highest = np.max(hl2)
        lowest = np.min(hl2)
        
        range_hl = highest - lowest
        if range_hl < 1e-10:
            range_hl = 1e-10
        
        # Normalize price
        x = (hl2[-1] - lowest) / range_hl
        x = np.clip(x, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        if i > 0:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    chop_12h = calculate_choppiness(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    hma_20 = calculate_hma(close, 20)
    hma_50 = calculate_hma(close, 50)
    
    # Calculate SMA for trend filter
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    stop_loss = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(200, n):
        # HTF trend filters
        hma_4h_valid = hma_4h_aligned[i] > 0
        hma_12h_valid = hma_12h_aligned[i] > 0
        
        price_above_4h_hma = close[i] > hma_4h_aligned[i] if hma_4h_valid else True
        price_below_4h_hma = close[i] < hma_4h_aligned[i] if hma_4h_valid else True
        
        price_above_12h_hma = close[i] > hma_12h_aligned[i] if hma_12h_valid else True
        price_below_12h_hma = close[i] < hma_12h_aligned[i] if hma_12h_valid else True
        
        # Regime detection from 12h
        is_ranging = chop_12h_aligned[i] > 50.0  # Loosened for more trades
        is_trending = chop_12h_aligned[i] < 45.0
        
        # 1h trend
        trend_bullish = hma_20[i] > hma_50[i] and close[i] > sma_200[i]
        trend_bearish = hma_20[i] < hma_50[i] and close[i] < sma_200[i]
        
        # CRSI signals (mean reversion)
        crsi_oversold = crsi[i] < 20  # Loosened from 10 for more trades
        crsi_overbought = crsi[i] > 80  # Loosened from 90 for more trades
        
        # Fisher Transform signals
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5 if i > 0 else False
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5 if i > 0 else False
        
        new_signal = 0.0
        
        # === MEAN REVERSION MODE (ranging market) ===
        if is_ranging:
            # Long: CRSI oversold + price above 4h HMA (bullish bias)
            if crsi_oversold and price_above_4h_hma:
                new_signal = SIZE_ENTRY
            
            # Short: CRSI overbought + price below 4h HMA (bearish bias)
            elif crsi_overbought and price_below_4h_hma:
                new_signal = -SIZE_ENTRY
        
        # === TREND FOLLOWING MODE (trending market) ===
        elif is_trending:
            # Long: Fisher long + trend bullish + 4h HMA support
            if fisher_long and trend_bullish and price_above_4h_hma:
                new_signal = SIZE_ENTRY
            
            # Short: Fisher short + trend bearish + 4h HMA resistance
            elif fisher_short and trend_bearish and price_below_4h_hma:
                new_signal = -SIZE_ENTRY
            
            # Pullback entry in trend
            elif trend_bullish and crsi[i] < 40 and price_above_4h_hma:
                new_signal = SIZE_ENTRY
            elif trend_bearish and crsi[i] > 60 and price_below_4h_hma:
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2*ATR from highest)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > stop_loss:
                stop_loss = current_stop
            
            # Check stoploss hit
            if close[i] < stop_loss:
                new_signal = SIZE_EXIT
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2*ATR from lowest)
            current_stop = lowest_close + 2.0 * atr[i]
            if stop_loss == 0.0 or current_stop < stop_loss:
                stop_loss = current_stop
            
            # Check stoploss hit
            if close[i] > stop_loss:
                new_signal = SIZE_EXIT
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stop_loss = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stop_loss = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            stop_loss = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals