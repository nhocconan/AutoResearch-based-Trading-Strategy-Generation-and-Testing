#!/usr/bin/env python3
"""
Experiment #004: 4h Mean Reversion with 1d HMA Trend Bias
Hypothesis: 4h timeframe captures multi-day swings better than lower TFs. 
Using 1d HMA as regime filter avoids counter-trend trades that destroyed #001-#003.
Entry on RSI extremes (mean reversion) but ONLY with daily trend direction.
This combines the best of both worlds: trend filter prevents whipsaw, RSI extremes catch pullbacks.
Key insight: BTC/ETH 2022 crash needs short bias, 2021 bull needs long bias. 1d HMA provides this.
Position sizing: 0.30 discrete, stoploss at 2.5*ATR trailing. Target 30-50 trades/year on 4h.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_meanrev_1d_hma_v1"
timeframe = "4h"
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

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    ema_1d_50 = calculate_ema(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for entries
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # Z-score for extreme mean reversion
    zscore = calculate_zscore(close, 20)
    
    # HMA on 4h for faster trend
    hma_4h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(ema_1d_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - this is the main regime filter
        bull_trend_1d = close[i] > hma_1d_aligned[i] and close[i] > ema_1d_50_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i] and close[i] < ema_1d_50_aligned[i]
        
        # 4h trend confirmation
        bull_trend_4h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_4h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter (only trade with 200 EMA direction)
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # RSI extreme conditions for mean reversion
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_oversold = rsi_7[i] < 25
        rsi_extreme_overbought = rsi_7[i] > 75
        
        # Bollinger Band extremes
        price_at_bb_lower = close[i] <= bb_lower[i] * 1.002
        price_at_bb_upper = close[i] >= bb_upper[i] * 0.998
        
        # Z-score extremes
        zscore_extreme_low = zscore[i] < -1.5
        zscore_extreme_high = zscore[i] > 1.5
        
        # HMA crossover on 4h
        hma_cross_long = i >= 1 and hma_4h[i] > ema_50[i] and hma_4h[i-1] <= ema_50[i-1]
        hma_cross_short = i >= 1 and hma_4h[i] < ema_50[i] and hma_4h[i-1] >= ema_50[i-1]
        
        # Price momentum (RSI rising/falling)
        rsi_rising = i >= 2 and rsi[i] > rsi[i-2]
        rsi_falling = i >= 2 and rsi[i] < rsi[i-2]
        
        # Price action: higher low for long, lower high for short
        higher_low = i >= 3 and low[i] > low[i-3]
        lower_high = i >= 3 and high[i] < high[i-3]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 1d bullish) ===
        if bull_trend_1d:
            # Primary: RSI oversold + price at BB lower (mean reversion in uptrend)
            if rsi_oversold and price_at_bb_lower and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: Z-score extreme low + RSI rising (reversal confirmation)
            elif zscore_extreme_low and rsi_rising and bull_trend_4h:
                new_signal = SIZE_BASE
            
            # Tertiary: RSI extreme oversold on 7-period (fast mean reversion)
            elif rsi_extreme_oversold and bull_trend_1d:
                new_signal = SIZE_HALF
            
            # HMA crossover with trend confirmation
            elif hma_cross_long and bull_trend_1d and above_200:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 1d bearish) ===
        elif bear_trend_1d:
            # Primary: RSI overbought + price at BB upper (mean reversion in downtrend)
            if rsi_overbought and price_at_bb_upper and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: Z-score extreme high + RSI falling (reversal confirmation)
            elif zscore_extreme_high and rsi_falling and bear_trend_4h:
                new_signal = -SIZE_BASE
            
            # Tertiary: RSI extreme overbought on 7-period (fast mean reversion)
            elif rsi_extreme_overbought and bear_trend_1d:
                new_signal = -SIZE_HALF
            
            # HMA crossover with trend confirmation
            elif hma_cross_short and bear_trend_1d and below_200:
                new_signal = -SIZE_HALF
        
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
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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