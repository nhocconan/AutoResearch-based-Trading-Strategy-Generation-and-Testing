#!/usr/bin/env python3
"""
Experiment #036: 1d Daily Trend-Follow with 1w HTF Regime Filter
Hypothesis: Daily timeframe reduces noise and fee drag. Weekly HMA provides clean regime bias.
Key insight: Previous 1d strategies failed due to overly strict entry conditions (0 trades).
This strategy uses LOOSE entry conditions: RSI 35-65 range, 2-6% pullback, multiple entry types.
Combines: trend-following (EMA/HMA), mean-reversion (BB/RSI), momentum (MACD).
Position sizing: 0.25-0.30 discrete levels with ATR trailing stops.
Why this might work: 1d has fewer whipsaws, weekly filter avoids counter-trend trades.
Must generate 10+ trades on train - entry conditions deliberately loosened vs failed experiments.
Timeframe: 1d (REQUIRED for exp#036), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trend_1w_hma_multi_entry_v1"
timeframe = "1d"
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

def calculate_bb(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD."""
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

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
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    bb_upper, bb_lower, bb_mid = calculate_bb(close, 20, 2.0)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    hma_21 = calculate_hma(close, 21)
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 1w trend bias (HTF) - main regime filter
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # 1d trend confirmation
        bull_trend_1d = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_1d = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # MACD momentum
        macd_bullish = macd_hist[i] > 0 and (i < 2 or macd_hist[i] > macd_hist[i-1])
        macd_bearish = macd_hist[i] < 0 and (i < 2 or macd_hist[i] < macd_hist[i-1])
        
        # RSI conditions - LOOSENED for more trades (35-65 range)
        rsi_pullback_long = 35 < rsi[i] < 60
        rsi_bounce_short = 40 < rsi[i] < 65
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Bollinger Band position
        bb_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        near_bb_lower = bb_pct < 0.15
        near_bb_upper = bb_pct > 0.85
        
        # Price pullback to EMA50 (2-6% range for more entries)
        pullback_long = False
        bounce_short = False
        if not np.isnan(ema_50[i]) and ema_50[i] > 0:
            pct_from_ema50 = (close[i] - ema_50[i]) / ema_50[i]
            pullback_long = -0.06 < pct_from_ema50 < -0.02
            bounce_short = 0.02 < pct_from_ema50 < 0.06
        
        # HMA crossover on 1d
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_21[i]) and not np.isnan(hma_21[i-1]):
            hma_cross_long = hma_21[i] > ema_50[i] and hma_21[i-1] <= ema_50[i-1]
            hma_cross_short = hma_21[i] < ema_50[i] and hma_21[i-1] >= ema_50[i-1]
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 5:
            higher_low = low[i] > min(low[i-3:i])
            lower_high = high[i] < max(high[i-3:i])
        
        new_signal = 0.0
        
        # === LONG ENTRIES (when 1w bullish) ===
        if bull_trend_1w:
            # Primary: Pullback to EMA50 with RSI confirmation
            if pullback_long and rsi_pullback_long and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: BB mean reversion in uptrend
            elif near_bb_lower and rsi_oversold and bull_trend_1d:
                new_signal = SIZE_BASE
            
            # Tertiary: HMA crossover with momentum
            elif hma_cross_long and macd_bullish:
                new_signal = SIZE_HALF
            
            # Momentum: Higher low with trend
            elif higher_low and bull_trend_1d and rsi[i] > 45:
                new_signal = SIZE_HALF
            
            # MACD crossover entry
            elif macd_bullish and rsi[i] > 50 and above_200:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (when 1w bearish) ===
        elif bear_trend_1w:
            # Primary: Bounce to EMA50 with RSI confirmation
            if bounce_short and rsi_bounce_short and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: BB mean reversion in downtrend
            elif near_bb_upper and rsi_overbought and bear_trend_1d:
                new_signal = -SIZE_BASE
            
            # Tertiary: HMA crossover with momentum
            elif hma_cross_short and macd_bearish:
                new_signal = -SIZE_HALF
            
            # Momentum: Lower high with trend
            elif lower_high and bear_trend_1d and rsi[i] < 55:
                new_signal = -SIZE_HALF
            
            # MACD crossover entry
            elif macd_bearish and rsi[i] < 50 and below_200:
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
            if lowest_close == 0.0 or close[i] < lowest_close:
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