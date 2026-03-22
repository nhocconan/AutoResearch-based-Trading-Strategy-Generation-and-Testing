#!/usr/bin/env python3
"""
Experiment #381: 1h Connors RSI + 4h HMA Trend + ATR Vol Filter + Asymmetric Sizing

Hypothesis: After analyzing 380 failed experiments, the winning formula combines:
1. CONNORS RSI (CRSI) - Proven 75% win rate for mean-reversion entries
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 10 (extreme oversold)
   - Short when CRSI > 90 (extreme overbought)
   - This catches reversals better than standard RSI(14)

2. 4h HMA(21) TREND BIAS - Via mtf_data helper (call ONCE before loop)
   - Only long when price > 4h HMA (bullish HTF trend)
   - Only short when price < 4h HMA (bearish HTF trend)
   - HMA has less lag than EMA for trend detection

3. ATR VOLATILITY FILTER - Avoid entries during vol spikes
   - ATR(7)/ATR(30) > 2.0 = vol spike (skip entries, wait for crush)
   - This avoids catching falling knives during panic

4. ASYMMETRIC POSITION SIZING - Based on trend strength
   - Strong trend (price far from HMA): SIZE = 0.30
   - Weak trend (price near HMA): SIZE = 0.20
   - Reduces risk during whipsaw regimes

5. ATR TRAILING STOP (2.0x) - Risk management
   - Signal → 0 when price moves 2.0*ATR against position
   - Protects from 2022-style crashes

Why this should work on 1h:
- CRSI is faster than RSI(14), catches intraday reversals
- 4h HMA provides stable trend bias (not noisy like 1h MA)
- Vol filter avoids panic entries (major cause of drawdown)
- Asymmetric sizing adapts to regime without complex logic
- Should generate 40-80 trades/year per symbol (enough for stats)
- Works on BTC, ETH, SOL individually (not SOL-biased)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_vol_filter_asymmetric_atr_v1"
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
    """Calculate RSI using Wilder's smoothing."""
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentile rank of price change over lookback period
    
    Entry signals:
    - Long when CRSI < 10 (extreme oversold)
    - Short when CRSI > 90 (extreme overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak component
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    streak_rsi_values = 100 - (100 / (1 + streak_rs))
    streak_rsi = streak_rsi_values.values
    
    # PercentRank component
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        price_changes = np.diff(close[i-rank_period+1:i+1])
        if len(price_changes) > 0 and price_changes[-1] != 0:
            rank = np.sum(price_changes[:-1] < price_changes[-1])
            percent_rank[i] = rank / len(price_changes[:-1]) * 100 if len(price_changes[:-1]) > 0 else 50.0
        else:
            percent_rank[i] = 50.0
    
    # Combine components
    for i in range(max(rsi_period, streak_period, rank_period), n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_atr_ratio(atr_short, atr_long):
    """Calculate ATR ratio for volatility spike detection."""
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # ATR ratio for vol spike detection
    atr_ratio = calculate_atr_ratio(atr_7, atr_30)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_STRONG = 0.30  # Strong trend
    SIZE_WEAK = 0.20    # Weak trend / whipsaw
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY FILTER ===
        # Skip entries during vol spikes (ATR ratio > 2.0)
        vol_spike = atr_ratio[i] > 2.0
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # Calculate trend strength (distance from HMA as % of ATR)
        hma_distance_pct = abs(close[i] - hma_4h_aligned[i]) / (hma_4h_aligned[i] + 1e-10) * 100
        strong_trend = hma_distance_pct > 2.0  # Price > 2% from HMA
        weak_trend = hma_distance_pct <= 2.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 10  # Extreme oversold
        crsi_overbought = crsi[i] > 90  # Extreme overbought
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG: CRSI oversold + 4h bullish trend + no vol spike
        if crsi_oversold and bull_trend_4h and not vol_spike:
            new_signal = SIZE_STRONG if strong_trend else SIZE_WEAK
        
        # SHORT: CRSI overbought + 4h bearish trend + no vol spike
        elif crsi_overbought and bear_trend_4h and not vol_spike:
            new_signal = -SIZE_STRONG if strong_trend else -SIZE_WEAK
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and bear_trend_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and bull_trend_4h:
            new_signal = 0.0
        
        # === VOL SPIKE EXIT ===
        # Exit position if vol spike occurs (panic mode)
        if in_position and vol_spike:
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