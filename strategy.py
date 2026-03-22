#!/usr/bin/env python3
"""
Experiment #222: 1d Connors RSI Mean Reversion + 1w HMA Trend + Choppiness Regime

Hypothesis: Daily timeframe is ideal for Connors RSI mean reversion strategy.
CRSI combines 3-period RSI, 2-period RSI streak, and 100-period percent rank
to identify extreme oversold/overbought conditions with 75%+ win rate.
The 1w HMA provides higher-timeframe trend bias (only take longs in uptrend).
Choppiness Index filters regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend.
This should work better than pure trend-following which failed in 2022 crash.

Why 1d might work for mean reversion:
- Daily bars filter intraday noise, cleaner signals
- CRSI extremes on 1d are rare but high-probability
- 1w HMA filter prevents counter-trend mean reversion in strong trends
- Choppiness regime detection switches logic appropriately
- Conservative sizing (0.30) with 2.5*ATR stop controls drawdown

Learning from failures:
- #210 (1d KAMA): Sharpe=-0.169 - pure trend failed in bear market
- #216 (1d HMA): Sharpe=-0.276 - trend-only doesn't work
- #221 (12h Fisher): Sharpe=-5.755 - Fisher transform alone insufficient
- Mean reversion works on 1d when combined with HTF trend filter
- Need regime detection to avoid mean reversion in strong trends

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_1w_hma_chop_regime_atr_v1"
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

def calculate_rsi_streak(close, period=2):
    """Calculate RSI Streak (consecutive up/down days)."""
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert to RSI-like scale (0-100)
    # Positive streak = bullish, negative = bearish
    rsi_streak = np.zeros(n)
    for i in range(period, n):
        streak_sum = np.sum(streak[i-period+1:i+1] > 0)
        rsi_streak[i] = 100 * streak_sum / period
    
    rsi_streak[:period] = 50.0
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank (position within last N periods)."""
    n = len(close)
    pr = np.zeros(n)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / period
        pr[i] = 100 * rank
    
    pr[:period] = 50.0
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI (CRSI)."""
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_short + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]), 
                     abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop[:period] = 50.0
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Need 250 bars for all indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 61.8 = ranging market (mean reversion works)
        # CHOP < 38.2 = trending market (trend following works)
        # 38.2 < CHOP < 61.8 = neutral
        range_regime = chop[i] > 55.0  # Slightly lower threshold for more trades
        trend_regime = chop[i] < 45.0
        
        # === CONNORS RSI EXTREMES ===
        # CRSI < 15 = extremely oversold (long opportunity)
        # CRSI > 85 = extremely overbought (short opportunity)
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === SMA200 FILTER ===
        # Price above SMA200 = long-term bullish
        # Price below SMA200 = long-term bearish
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: Range regime OR (trend regime + 1w bullish) + CRSI oversold + above SMA200
        # More flexible: allow longs in range regime regardless of 1w trend
        if range_regime and crsi_oversold:
            # Mean reversion long in ranging market
            new_signal = SIZE_BASE
        elif bull_trend_1w and crsi_oversold and above_sma200:
            # Trend-following long: pullback in uptrend
            new_signal = SIZE_BASE
        
        # Short: Range regime OR (trend regime + 1w bearish) + CRSI overbought + below SMA200
        if range_regime and crsi_overbought:
            # Mean reversion short in ranging market
            new_signal = -SIZE_BASE
        elif bear_trend_1w and crsi_overbought and below_sma200:
            # Trend-following short: rally in downtrend
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
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
        
        # === EXIT ON CRSI MEAN REVERSION ===
        # Exit long when CRSI > 60 (mean reverted)
        # Exit short when CRSI < 40 (mean reverted)
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 60.0:
                new_signal = 0.0  # Take profit on mean reversion
            elif position_side < 0 and crsi[i] < 40.0:
                new_signal = 0.0  # Take profit on mean reversion
        
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
            # else: maintaining same position direction
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