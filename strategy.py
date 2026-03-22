#!/usr/bin/env python3
"""
Experiment #045: 1h Multi-Timeframe Trend-Follow with 4h HMA Bias + Z-Score Filter
Hypothesis: 1h timeframe captures shorter-term momentum while 4h HMA provides regime filter.
Adding Z-score filter to avoid extreme entries (proven in baseline Sharpe=5.4 strategy).
Key insight: Previous 12h/1d strategies had too few trades. 1h should generate more entries.
Position sizing: 0.25-0.30 discrete levels, stoploss at 2.5*ATR.
Timeframe: 1h (REQUIRED for exp#045), HTF: 4h via mtf_data helper.
Why this might work: Combines proven 4h HMA trend + 1h RSI pullback + Z-score (baseline formula).
Must generate 10+ trades on train, 3+ on test - entry conditions loosened vs strict filters.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_zscore_v1"
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

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter."""
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    zscore = calculate_zscore(close, 20)
    
    # HMA on 1h for faster trend
    hma_1h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss (persistent state)
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        new_signal = 0.0
        
        # === STOPLOSS LOGIC FIRST (Rule 6) ===
        # Check if existing position should be stopped out
        if position_side > 0:  # Long position active
            if close[i] > highest_close:
                highest_close = close[i]
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            if close[i] < trailing_stop:
                new_signal = 0.0  # Force exit
        
        elif position_side < 0:  # Short position active
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            if close[i] > trailing_stop:
                new_signal = 0.0  # Force exit
        
        # === ENTRY SIGNALS (only if not stopped out) ===
        if new_signal != 0.0:
            pass  # Already forced to 0 by stoploss
        else:
            # 4h trend bias (HTF) - main regime filter
            bull_trend_4h = close[i] > hma_4h_aligned[i]
            bear_trend_4h = close[i] < hma_4h_aligned[i]
            
            # 1h trend confirmation
            bull_trend_1h = ema_21[i] > ema_50[i]
            bear_trend_1h = ema_21[i] < ema_50[i]
            
            # Long-term trend filter
            above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
            below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
            
            # RSI conditions - LOOSENED for more trades
            rsi_pullback_long = 35 < rsi[i] < 60
            rsi_bounce_short = 40 < rsi[i] < 65
            rsi_oversold = rsi[i] < 40
            rsi_overbought = rsi[i] > 60
            
            # Z-score filter - avoid extreme entries
            zscore_neutral = abs(zscore[i]) < 2.0
            
            # HMA crossover on 1h
            hma_cross_long = False
            hma_cross_short = False
            if i >= 1 and not np.isnan(hma_1h[i]) and not np.isnan(hma_1h[i-1]):
                hma_cross_long = hma_1h[i] > ema_50[i] and hma_1h[i-1] <= ema_50[i-1]
                hma_cross_short = hma_1h[i] < ema_50[i] and hma_1h[i-1] >= ema_50[i-1]
            
            # Price pullback to EMA21
            price_near_ema21_long = close[i] <= ema_21[i] * 1.015 and close[i] >= ema_21[i] * 0.985
            price_near_ema21_short = close[i] >= ema_21[i] * 0.985 and close[i] <= ema_21[i] * 1.015
            
            # Price action: higher low for long, lower high for short
            higher_low = False
            lower_high = False
            if i >= 3:
                higher_low = low[i] > low[i-3]
                lower_high = high[i] < high[i-3]
            
            # === LONG ENTRIES (only when 4h bullish) ===
            if bull_trend_4h and position_side <= 0:
                # Primary: Pullback to EMA21 with RSI confirmation
                if price_near_ema21_long and rsi_pullback_long and above_200:
                    new_signal = SIZE_BASE
                
                # Secondary: HMA crossover with 4h confirmation
                elif hma_cross_long and bull_trend_4h and bull_trend_1h:
                    new_signal = SIZE_BASE
                
                # Tertiary: RSI oversold bounce in uptrend
                elif rsi_oversold and bull_trend_1h and zscore_neutral:
                    new_signal = SIZE_HALF
                
                # Momentum: Higher low with trend
                elif higher_low and bull_trend_1h and rsi[i] > 45:
                    new_signal = SIZE_HALF
            
            # === SHORT ENTRIES (only when 4h bearish) ===
            elif bear_trend_4h and position_side >= 0:
                # Primary: Bounce to EMA21 with RSI confirmation
                if price_near_ema21_short and rsi_bounce_short and below_200:
                    new_signal = -SIZE_BASE
                
                # Secondary: HMA crossover with 4h confirmation
                elif hma_cross_short and bear_trend_4h and bear_trend_1h:
                    new_signal = -SIZE_BASE
                
                # Tertiary: RSI overbought rejection in downtrend
                elif rsi_overbought and bear_trend_1h and zscore_neutral:
                    new_signal = -SIZE_HALF
                
                # Momentum: Lower high with trend
                elif lower_high and bear_trend_1h and rsi[i] < 55:
                    new_signal = -SIZE_HALF
        
        # === UPDATE POSITION TRACKING AFTER SIGNAL ===
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and position_side == 0:
            # Enter new position from flat
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and position_side != 0 and np.sign(new_signal) != position_side:
            # Reverse position
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and position_side != 0:
            # Exit position
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals