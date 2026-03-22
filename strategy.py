#!/usr/bin/env python3
"""
Experiment #047: 12h Trend-Follow with 1d HMA Regime + RSI Pullback Entries
Hypothesis: 12h timeframe captures intermediate trends while avoiding 1h/4h whipsaws.
1d HMA provides clean regime filter (bull/bear), 12h EMA pullbacks give entry timing.
Key insight from failures: Too many conflicting filters = 0 trades. Simplify entry logic.
Position sizing: 0.25 base, 0.15 half-size for weaker signals. ATR trailing stop 2.5x.
Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
Why this might beat Sharpe=0.162: Looser RSI conditions (30-65 range), fewer filters,
better stoploss tracking, discrete signal levels to minimize fee churn.
Must generate 10+ trades on train, 3+ on test - entry conditions deliberately loosened.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trend_1d_hma_rsi_pullback_v3"
timeframe = "12h"
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

def calculate_supertrend(high, low, close, period=10, mult=3.0):
    """Calculate Supertrend for trend direction."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    
    supertrend = np.zeros(len(close))
    trend = np.ones(len(close))
    
    supertrend[0] = upper[0]
    for i in range(1, len(close)):
        if trend[i-1] == 1:
            if close[i] < lower[i]:
                trend[i] = -1
                supertrend[i] = upper[i]
            else:
                trend[i] = 1
                supertrend[i] = max(lower[i], supertrend[i-1])
        else:
            if close[i] > upper[i]:
                trend[i] = 1
                supertrend[i] = lower[i]
            else:
                trend[i] = -1
                supertrend[i] = min(upper[i], supertrend[i-1])
    
    return supertrend, trend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    
    # Supertrend for trend confirmation
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 12h trend confirmation
        bull_trend_12h = ema_21[i] > ema_50[i]
        bear_trend_12h = ema_21[i] < ema_50[i]
        
        # Supertrend confirmation
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # RSI conditions - LOOSENED for more trades (key fix)
        rsi_pullback_long = 30 < rsi[i] < 65
        rsi_bounce_short = 35 < rsi[i] < 70
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # Price pullback to EMA21 (entry zone)
        price_near_ema21_long = close[i] <= ema_21[i] * 1.03 and close[i] >= ema_21[i] * 0.97
        price_near_ema21_short = close[i] >= ema_21[i] * 0.97 and close[i] <= ema_21[i] * 1.03
        
        # Price pullback to EMA50 (deeper entry)
        price_near_ema50_long = close[i] <= ema_50[i] * 1.03 and close[i] >= ema_50[i] * 0.97
        price_near_ema50_short = close[i] >= ema_50[i] * 0.97 and close[i] <= ema_50[i] * 1.03
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 3:
            higher_low = low[i] > low[i-3]
            lower_high = high[i] < high[i-3]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 1d bullish) ===
        if bull_trend_1d:
            # Primary: Pullback to EMA21 with RSI confirmation + 12h trend
            if price_near_ema21_long and rsi_pullback_long and bull_trend_12h:
                new_signal = SIZE_BASE
            
            # Secondary: Deeper pullback to EMA50 with RSI oversold
            elif price_near_ema50_long and rsi_oversold and above_200:
                new_signal = SIZE_BASE
            
            # Tertiary: Supertrend flip bullish with 1d confirmation
            elif st_bullish and bull_trend_1d and rsi[i] > 40:
                new_signal = SIZE_HALF
            
            # Momentum: Higher low with trend alignment
            elif higher_low and bull_trend_12h and st_bullish and rsi[i] > 45:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 1d bearish) ===
        elif bear_trend_1d:
            # Primary: Bounce to EMA21 with RSI confirmation + 12h trend
            if price_near_ema21_short and rsi_bounce_short and bear_trend_12h:
                new_signal = -SIZE_BASE
            
            # Secondary: Deeper bounce to EMA50 with RSI overbought
            elif price_near_ema50_short and rsi_overbought and below_200:
                new_signal = -SIZE_BASE
            
            # Tertiary: Supertrend flip bearish with 1d confirmation
            elif st_bearish and bear_trend_1d and rsi[i] < 60:
                new_signal = -SIZE_HALF
            
            # Momentum: Lower high with trend alignment
            elif lower_high and bear_trend_12h and st_bearish and rsi[i] < 55:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss - trailing
        if position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss - trailing
        if position_side < 0:
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
            # New entry
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            # Position flip
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            # Exit position
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals