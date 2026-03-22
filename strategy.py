#!/usr/bin/env python3
"""
Experiment #003: 1h Mean Reversion + 4h HMA Trend + Vol Spike Filter

Hypothesis: BTC/ETH fail with pure trend following but excel with mean reversion when 
filtered by higher timeframe trend direction. This strategy combines:

1. Connors RSI (CRSI) - Mean reversion signal with proven edge
   CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long when CRSI < 20, Short when CRSI > 80 (looser than extreme for more trades)

2. 4h HMA Trend Filter - Via mtf_data helper (NO manual resampling)
   Only take longs when price > 4h HMA (bull regime)
   Only take shorts when price < 4h HMA (bear regime)

3. Volatility Spike Filter - ATR(7)/ATR(30) ratio > 1.5
   Enter when vol spike indicates panic/reversal opportunity

4. Bollinger Band Confirmation - Price at band extremes for additional filter

5. Asymmetric Logic - Different thresholds for bull vs bear regimes
   Bull: More aggressive longs, conservative shorts only at extremes
   Bear: More aggressive shorts, conservative longs only at extremes

6. Conservative Sizing - 0.25 base, 0.35 max, 2.5*ATR stoploss
   Prevents blowup during 2022-style crashes (77% BTC drawdown)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data.get_htf_data() - called ONCE before loop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_4h_hma_volspike_v1"
timeframe = "1h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using EMA smoothing."""
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

def calculate_streak_rsi(close, period=2):
    """Calculate RSI of consecutive up/down streaks for Connors RSI."""
    n = len(close)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on absolute streak values
    streak_rsi = calculate_rsi(np.abs(streak), period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank for Connors RSI."""
    n = len(close)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(period, n):
        window = close[i-period:i]
        current = close[i]
        rank = np.sum(window < current)
        percent_rank[i] = rank / period * 100
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI."""
    rsi = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_streak_rsi(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi + streak_rsi + percent_rank) / 3
    return crsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align 4h HMA to 1h timeframe (CRITICAL - Rule 2, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    crsi = calculate_crsi(close, 3, 2, 100)
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.5)
    
    # Volatility spike ratio
    vol_spike_ratio = np.zeros(n)
    mask = atr_30 > 0
    vol_spike_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_50[i]) or np.isnan(ema_200[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        bull_regime = close[i] > hma_4h_aligned[i]
        bear_regime = close[i] < hma_4h_aligned[i]
        
        # Volatility spike (panic/reversal opportunity)
        vol_spike = vol_spike_ratio[i] > 1.5
        
        # Connors RSI extremes - LOOSENED for more trades
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_oversold = crsi[i] < 15
        crsi_extreme_overbought = crsi[i] > 85
        
        # Bollinger extremes
        price_below_lower = close[i] < bb_lower[i] * 1.01
        price_above_upper = close[i] > bb_upper[i] * 0.99
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === BULL REGIME (price > 4h HMA): Favor longs ===
        if bull_regime:
            # Strong long: CRSI oversold + vol spike + price at BB lower
            if crsi_extreme_oversold and vol_spike and price_below_lower:
                new_signal = SIZE_MAX
            # Moderate long: CRSI oversold in bull regime + EMA confirmation
            elif crsi_oversold and bull_regime and ema_bullish:
                new_signal = SIZE_BASE
            # Conservative long: CRSI oversold in bull regime (no vol spike needed)
            elif crsi_oversold and bull_regime:
                new_signal = SIZE_BASE
            # Conservative short: Only at extreme overbought + BB upper
            elif crsi_extreme_overbought and price_above_upper:
                new_signal = -SIZE_BASE
        
        # === BEAR REGIME (price < 4h HMA): Favor shorts ===
        elif bear_regime:
            # Strong short: CRSI overbought + vol spike + price at BB upper
            if crsi_extreme_overbought and vol_spike and price_above_upper:
                new_signal = -SIZE_MAX
            # Moderate short: CRSI overbought in bear regime + EMA confirmation
            elif crsi_overbought and bear_regime and ema_bearish:
                new_signal = -SIZE_BASE
            # Conservative short: CRSI overbought in bear regime
            elif crsi_overbought and bear_regime:
                new_signal = -SIZE_BASE
            # Conservative long: Only at extreme oversold + BB lower
            elif crsi_extreme_oversold and price_below_lower:
                new_signal = SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals