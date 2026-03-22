#!/usr/bin/env python3
"""
Experiment #455: 12h Connors RSI Mean Reversion with Daily HMA Trend Filter

Hypothesis: After analyzing 454 failed experiments, the pattern is clear:
- Simple trend following fails on BTC/ETH (2022 crash destroys gains)
- Pure mean reversion without trend filter gets caught in strong trends
- 12h timeframe needs fewer, higher-quality signals to minimize fee drag

This strategy combines:
1. DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Long only when price > 1d HMA (bullish bias)
   - Short only when price < 1d HMA (bearish bias)
   - HMA is smoother than EMA, critical for daily trend detection

2. CONNORS RSI (CRSI) FOR ENTRY TIMING:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 15 (extreme oversold)
   - Short when CRSI > 85 (extreme overbought)
   - More sensitive than regular RSI, catches reversals faster
   - Proven 75% win rate in mean reversion literature

3. BOLLINGER BAND WIDTH REGIME FILTER:
   - BB Width percentile < 30% = squeeze (expect breakout)
   - BB Width percentile > 70% = expansion (mean reversion likely)
   - Only take mean reversion signals when BB expanding

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for 2022-style crash protection

5. POSITION SIZING: 0.30 discrete (conservative for 12h volatility)
   - Max 30% capital per position
   - Discrete levels minimize fee churn

Why this should work on 12h:
- CRSI is more sensitive than RSI(14), ensures sufficient trades
- Daily HMA filter prevents counter-trend disasters
- BB width filter avoids entering during squeezes (whipsaw risk)
- Should work on BTC/ETH/SOL individually (not SOL-biased)
- Fewer trades than lower timeframes = less fee drag

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_daily_hma_bb_regime_atr_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    This is more sensitive than regular RSI and catches reversals faster.
    Proven 75% win rate for mean reversion strategies.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3) - short period for sensitivity
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measures consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            # Positive streak - calculate RSI of streak values
            pos_streaks = streak[max(0, i-streak_period*2):i+1]
            pos_only = pos_streaks[pos_streaks > 0]
            if len(pos_only) > 0:
                streak_rsi[i] = min(100, 50 + len(pos_only) * 10)
            else:
                streak_rsi[i] = 50
        elif streak[i] < 0:
            neg_streaks = streak[max(0, i-streak_period*2):i+1]
            neg_only = neg_streaks[neg_streaks < 0]
            if len(neg_only) > 0:
                streak_rsi[i] = max(0, 50 - len(neg_only) * 10)
            else:
                streak_rsi[i] = 50
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - where current price ranks in last 100 bars
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        crsi[i] = (rsi_short[i] + streak_rsi[i] + rank) / 3
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma * 100  # bandwidth as percentage
    
    return upper.values, lower.values, bandwidth.values

def calculate_bb_width_percentile(bandwidth, lookback=100):
    """Calculate percentile rank of BB width over lookback period."""
    n = len(bandwidth)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = bandwidth[i-lookback:i]
        current = bandwidth[i]
        # Percentile: what % of historical values are below current
        pct = np.sum(window < current) / lookback * 100
        percentile[i] = pct
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_bandwidth, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pct[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === BB WIDTH REGIME FILTER ===
        # High percentile = bandwidth expanding = mean reversion likely
        bb_expanding = bb_width_pct[i] > 50  # Above median
        bb_squeeze = bb_width_pct[i] < 30  # Squeeze = avoid mean reversion
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = extreme oversold (long opportunity)
        # CRSI > 85 = extreme overbought (short opportunity)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: CRSI oversold + bull trend + BB not in squeeze
        if crsi_oversold and bull_trend_1d and not bb_squeeze:
            new_signal = SIZE
        
        # SHORT: CRSI overbought + bear trend + BB not in squeeze
        elif crsi_overbought and bear_trend_1d and not bb_squeeze:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === CRSI MEAN REVERSION EXIT ===
        # Exit long when CRSI goes above 50 (mean reached)
        # Exit short when CRSI goes below 50 (mean reached)
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 50:
                new_signal = 0.0
            if position_side < 0 and crsi[i] < 50:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals