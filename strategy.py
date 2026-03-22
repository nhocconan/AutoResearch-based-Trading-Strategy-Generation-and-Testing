#!/usr/bin/env python3
"""
Experiment #001: 15m Multi-Timeframe Mean Reversion with 4h Trend Bias
Hypothesis: 15m timeframe captures more mean reversion opportunities than 30m/1h.
Combined with 4h HMA trend filter, this avoids counter-trend trades during strong moves.
Key components:
1. Connors RSI (CRSI) for mean reversion entries - proven 75% win rate in research
2. Choppiness Index (CHOP) for regime detection - range vs trend filtering
3. 4h HMA aligned via mtf_data helper for trend bias (no look-ahead)
4. Fisher Transform for reversal confirmation
5. ATR trailing stop at 2.0*ATR for risk management
Position sizing: 0.25 base, 0.15 half - discrete levels to minimize fee churn
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data.get_htf_data()
Why 15m: More trades than 30m/1h, better for mean reversion, captures intraday swings
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_chop_4h_hma_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 and close[i-1] >= close[i-2] else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 and close[i-1] <= close[i-2] else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak_abs[i] == 0:
            streak_rsi[i] = 50
        else:
            # Normalize streak to 0-100 scale
            max_streak = max(np.max(streak_abs[max(0, i-streak_period*2):i+1]), 1)
            streak_rsi[i] = 50 + 50 * (streak[i] / max_streak)
    
    # Percent Rank - where current price ranks in last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window < close[i]) / rank_period
        percent_rank[i] = rank * 100
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_fast) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_fast[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(atr[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    Catches reversals in bear/range markets.
    """
    n = len(close) if 'close' in dir() else len(high)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    # Use typical price
    typical = (high + low + high) / 3  # Simplified
    
    for i in range(period, n):
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        range_val = highest - lowest
        
        if range_val > 0:
            normalized = 0.66 * ((typical[i] - lowest) / range_val - 0.5)
            normalized = np.clip(normalized, -0.99, 0.99)
            
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            if i > period:
                fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # Choppiness regime detection
        is_range = chop[i] > 55  # Range market (mean revert)
        is_trend = chop[i] < 45  # Trending market (trend follow)
        
        # Long-term trend filter
        above_200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # CRSI mean reversion signals
        crsi_oversold = crsi[i] < 15  # Extreme oversold
        crsi_overbought = crsi[i] > 85  # Extreme overbought
        crsi_neutral_low = 15 <= crsi[i] <= 35
        crsi_neutral_high = 65 <= crsi[i] <= 85
        
        # Bollinger Band position
        price_near_bb_lower = close[i] <= bb_lower[i] * 1.005
        price_near_bb_upper = close[i] >= bb_upper[i] * 0.995
        price_at_bb_mid = abs(close[i] - bb_mid[i]) < atr[i] * 0.5
        
        # Fisher Transform reversal signals
        fisher_cross_up = fisher_signal[i] < -1.5 and fisher[i] >= -1.5 if not np.isnan(fisher_signal[i]) else False
        fisher_cross_down = fisher_signal[i] > 1.5 and fisher[i] <= 1.5 if not np.isnan(fisher_signal[i]) else False
        
        # EMA alignment for trend confirmation
        ema_bull = ema_21[i] > ema_50[i] and close[i] > ema_21[i]
        ema_bear = ema_21[i] < ema_50[i] and close[i] < ema_21[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: CRSI oversold + 4h bullish + range market (mean reversion)
        if crsi_oversold and bull_trend_4h and is_range:
            new_signal = SIZE_BASE
        
        # Secondary: CRSI oversold + Fisher cross up + above 200 EMA
        elif crsi_oversold and fisher_cross_up and above_200:
            new_signal = SIZE_BASE
        
        # Tertiary: Price at BB lower + 4h bullish + RSI < 40
        elif price_near_bb_lower and bull_trend_4h and rsi[i] < 40:
            new_signal = SIZE_HALF
        
        # Momentum: EMA bull + 4h bullish + pullback to EMA21
        elif ema_bull and bull_trend_4h and close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.99:
            new_signal = SIZE_HALF
        
        # === SHORT ENTRIES ===
        # Primary: CRSI overbought + 4h bearish + range market (mean reversion)
        if crsi_overbought and bear_trend_4h and is_range:
            new_signal = -SIZE_BASE
        
        # Secondary: CRSI overbought + Fisher cross down + below 200 EMA
        elif crsi_overbought and fisher_cross_down and below_200:
            new_signal = -SIZE_BASE
        
        # Tertiary: Price at BB upper + 4h bearish + RSI > 60
        elif price_near_bb_upper and bear_trend_4h and rsi[i] > 60:
            new_signal = -SIZE_HALF
        
        # Momentum: EMA bear + 4h bearish + bounce to EMA21
        elif ema_bear and bear_trend_4h and close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.01:
            new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals