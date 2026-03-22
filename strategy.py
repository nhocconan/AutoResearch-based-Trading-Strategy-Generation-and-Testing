#!/usr/bin/env python3
"""
Experiment #123: 1h Connors RSI + 4h HMA Trend + Choppiness Regime Filter

Hypothesis: After 122 failed experiments, combining proven mean-reversion (Connors RSI)
with multi-timeframe trend filter and regime detection should work better than pure trend:
- Connors RSI (CRSI) has 75% win rate on pullbacks when aligned with HTF trend
- 4h HMA(21) provides stable trend bias (avoids counter-trend trades in 2022 crash)
- Choppiness Index (CHOP) detects regime: >61.8 = range (mean revert), <38.2 = trend
- ATR(14) trailing stop at 2.5*ATR protects against reversals
- 1h timeframe generates enough trades (50-100/year) while filtering noise

Why this might beat mtf_4h_kama_1d_hma_adx_atr_v1 (Sharpe=0.478):
- CRSI catches pullbacks in trends better than simple RSI
- CHOP regime filter avoids mean-reversion losses during strong trends
- 4h HMA filter is smoother than 1d for 1h entries (better alignment)
- Conservative position sizing (0.25/0.35) limits drawdown

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_chop_regime_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven 75% win rate on mean-reversion entries when aligned with trend.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    if n < rank_period + 10:
        return crsi
    
    # RSI(3) component
    rsi3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI(2)
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.inf)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi.fillna(50.0).values
    
    # PercentRank component (rank of today's return vs last 100 days)
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1] * 100
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        if len(window) > 0:
            percent_rank[i] = np.sum(window < returns[i]) / len(window) * 100
    
    # Combine components
    valid_mask = ~np.isnan(rsi3) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi3[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    if n < period:
        return chop
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = np.sum(np.abs(high[i-period+1:i+1] - low[i-period+1:i+1]))
        
        if highest_high > lowest_low and atr_sum > 0:
            chop[i] = 100 * np.log10((highest_high - lowest_low) / atr_sum) / np.log10(period)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    sma200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
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
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma200[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h SMA200 = local trend filter
        bull_trend_1h = close[i] > sma200[i]
        bear_trend_1h = close[i] < sma200[i]
        
        # Combined trend bias (both HTF and LTF agree)
        strong_bull = bull_trend_4h and bull_trend_1h
        strong_bear = bear_trend_4h and bear_trend_1h
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = ranging (mean reversion)
        # CHOP < 38.2 = trending (breakout/pullback)
        is_ranging = chop[i] > 55.0  # Slightly lower threshold for more trades
        is_trending = chop[i] < 45.0  # Slightly higher threshold for more trades
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        if strong_bull:
            if is_ranging:
                # Mean reversion in uptrend: buy deep pullback
                if crsi[i] < 35:  # Relaxed from 20 for more trades
                    new_signal = SIZE_STRONG
                elif crsi[i] < 45:
                    new_signal = SIZE_BASE
            elif is_trending:
                # Trend pullback: buy moderate pullback
                if crsi[i] < 40:
                    new_signal = SIZE_STRONG
                elif crsi[i] < 50:
                    new_signal = SIZE_BASE
            else:
                # Neutral regime: use moderate CRSI
                if crsi[i] < 35:
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        elif strong_bear:
            if is_ranging:
                # Mean reversion in downtrend: sell deep rally
                if crsi[i] > 65:  # Relaxed from 80 for more trades
                    new_signal = -SIZE_STRONG
                elif crsi[i] > 55:
                    new_signal = -SIZE_BASE
            elif is_trending:
                # Trend pullback: sell moderate rally
                if crsi[i] > 60:
                    new_signal = -SIZE_STRONG
                elif crsi[i] > 50:
                    new_signal = -SIZE_BASE
            else:
                # Neutral regime: use moderate CRSI
                if crsi[i] > 65:
                    new_signal = -SIZE_BASE
        else:
            # No strong trend bias - only trade extreme CRSI in ranging market
            if is_ranging:
                if crsi[i] < 25:
                    new_signal = SIZE_BASE
                elif crsi[i] > 75:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals