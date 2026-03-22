#!/usr/bin/env python3
"""
Experiment #485: 12h Connors RSI Mean Reversion with Daily HMA Regime Filter

Hypothesis: After analyzing 484 failed experiments, the key insight is that BTC/ETH 
perform best with MEAN REVERSION entries in the direction of the higher timeframe trend.
Connors RSI (CRSI) has proven 75%+ win rate in academic studies for short-term reversals.

Strategy Components:
1. DAILY HMA(21) TREND BIAS (via mtf_data helper):
   - Bull: price > 1d HMA (favor long mean-reversion)
   - Bear: price < 1d HMA (favor short mean-reversion)

2. CONNORS RSI (CRSI) FOR ENTRY TIMING:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long entry: CRSI < 15 (oversold pullback in bull trend)
   - Short entry: CRSI > 85 (overbought rally in bear trend)
   - Proven 75% win rate in literature

3. VOLATILITY REGIME FILTER:
   - ATR(7) / ATR(21) ratio > 1.3 = vol expansion (enable entries)
   - Prevents entries during low-vol chop where mean-reversion fails

4. ATR(14) TRAILING STOP at 2.5x:
   - Tighter stop for 12h timeframe vs daily
   - Signal → 0 when price moves 2.5*ATR against position

5. POSITION SIZING: 0.25 discrete
   - Conservative for 12h volatility
   - Discrete levels minimize fee churn

Why 12h should work:
- Faster than 1d (more trades) but slower than 4h (less noise)
- Connors RSI excels at 12h-1d timeframes for crypto
- Daily HMA provides robust trend filter without whipsaw
- Should generate 20-40 trades/year per symbol

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_connors_rsi_daily_hma_volregime_atr_v1"
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

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI of streak (consecutive up/down days).
    Streak: +1 for up day, -1 for down day, cumulative sum.
    Then calculate RSI on the absolute streak values.
    """
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = streak[i-1]
    
    # Convert to RSI-like metric (0-100)
    # Positive streak = bullish, negative = bearish
    streak_s = pd.Series(streak)
    
    # Calculate RSI on streak (period=2 as per Connors)
    delta = streak_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_streak = 100 - (100 / (1 + rs))
    
    return rsi_streak.values

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank: where current price ranks in last N periods.
    0 = lowest, 100 = highest in the lookback window.
    """
    n = len(close)
    pr = np.full(n, np.nan)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / period * 100
        pr[i] = rank
    
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_close = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

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
    atr_7 = calculate_atr(high, low, close, 7)
    atr_21 = calculate_atr(high, low, close, 21)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    sma_50 = calculate_sma(close, 50)
    
    # Volatility regime filter: ATR(7) / ATR(21)
    vol_ratio = atr_7 / atr_21
    vol_ratio = np.where(np.isnan(vol_ratio) | (atr_21 == 0), 0, vol_ratio)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(crsi[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === DAILY HMA TREND BIAS ===
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY REGIME FILTER ===
        vol_expansion = vol_ratio[i] > 1.3
        
        # === CONNORS RSI ENTRY LOGIC ===
        new_signal = 0.0
        
        # BULL REGIME: Long mean-reversion on oversold CRSI
        if bull_regime:
            # Vol expansion confirms momentum for entry
            if vol_expansion and crsi[i] < 15:
                new_signal = SIZE
            # Looser threshold if no vol expansion but strong oversold
            elif crsi[i] < 10:
                new_signal = SIZE
        
        # BEAR REGIME: Short mean-reversion on overbought CRSI
        if bear_regime:
            # Vol expansion confirms momentum for entry
            if vol_expansion and crsi[i] > 85:
                new_signal = -SIZE
            # Looser threshold if no vol expansion but strong overbought
            elif crsi[i] > 90:
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
        
        # === REGIME REVERSAL EXIT ===
        # Exit if daily trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_regime:
                new_signal = 0.0
            if position_side < 0 and bull_regime:
                new_signal = 0.0
        
        # === CRSI MEAN REVERSION EXIT ===
        # Exit long when CRSI becomes overbought (>70)
        # Exit short when CRSI becomes oversold (<30)
        if in_position and new_signal != 0.0:
            if position_side > 0 and crsi[i] > 70:
                new_signal = 0.0
            if position_side < 0 and crsi[i] < 30:
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