#!/usr/bin/env python3
"""
Experiment #007: 15m Multi-Timeframe Supertrend + RSI Pullback with 4h HMA Bias
Hypothesis: 15m timeframe captures intraday momentum while 4h HMA provides cleaner trend bias than 1h.
Key insight: Previous 15m strategy (#001) failed due to complex Choppiness filter. This uses simpler logic:
4h HMA for regime, 15m Supertrend for momentum, RSI pullback (35-65 range) for entries.
Why 15m: More trades than 12h/1d, captures intraday moves, but needs HTF filter to avoid whipsaws.
Position sizing: 0.20-0.25 discrete (smaller than 12h due to more trades/fees), 2*ATR stoploss.
Entry conditions LOOSENED vs failed experiments to ensure 10+ trades on train, 3+ on test.
Timeframe: 15m (REQUIRED for exp#007), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_rsi_4h_hma_v1"
timeframe = "15m"
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
    trend = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
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
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_50 = calculate_sma(close, 50)
    
    # Supertrend for trend confirmation (15m)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    
    # HMA on 15m for faster trend signal
    hma_15m = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    # Smaller than 12h strategy due to more trades on 15m
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 15m trend confirmation
        bull_trend_15m = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_15m = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Supertrend confirmation
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # RSI conditions - LOOSENED for more trades (35-65 pullback range)
        rsi_pullback_long = 35 < rsi[i] < 65
        rsi_pullback_short = 35 < rsi[i] < 65
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # HMA crossover on 15m
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_15m[i]) and not np.isnan(hma_15m[i-1]):
            hma_cross_long = hma_15m[i] > ema_50[i] and hma_15m[i-1] <= ema_50[i-1]
            hma_cross_short = hma_15m[i] < ema_50[i] and hma_15m[i-1] >= ema_50[i-1]
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.015 and close[i] >= ema_21[i] * 0.985
        price_near_ema21_short = close[i] >= ema_21[i] * 0.985 and close[i] <= ema_21[i] * 1.015
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 3:
            higher_low = low[i] > low[i-3]
            lower_high = high[i] < high[i-3]
        
        # EMA21 slope
        ema21_slope_up = False
        ema21_slope_down = False
        if i >= 5 and not np.isnan(ema_21[i]) and not np.isnan(ema_21[i-5]):
            ema21_slope_up = ema_21[i] > ema_21[i-5]
            ema21_slope_down = ema_21[i] < ema_21[i-5]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 4h bullish) ===
        if bull_trend_4h:
            # Primary: Pullback to EMA21 with RSI confirmation + Supertrend
            if price_near_ema21_long and rsi_pullback_long and st_bullish:
                new_signal = SIZE_BASE
            
            # Secondary: HMA crossover with 4h confirmation
            elif hma_cross_long and bull_trend_4h and above_200:
                new_signal = SIZE_BASE
            
            # Tertiary: RSI oversold bounce in uptrend
            elif rsi_oversold and bull_trend_15m and ema21_slope_up:
                new_signal = SIZE_HALF
            
            # Momentum: Higher low with trend + Supertrend
            elif higher_low and bull_trend_15m and st_bullish:
                new_signal = SIZE_HALF
            
            # Breakout: Price above EMA50 with momentum
            elif close[i] > ema_50[i] and rsi[i] > 50 and st_bullish:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 4h bearish) ===
        elif bear_trend_4h:
            # Primary: Bounce to EMA21 with RSI confirmation + Supertrend
            if price_near_ema21_short and rsi_pullback_short and st_bearish:
                new_signal = -SIZE_BASE
            
            # Secondary: HMA crossover with 4h confirmation
            elif hma_cross_short and bear_trend_4h and below_200:
                new_signal = -SIZE_BASE
            
            # Tertiary: RSI overbought rejection in downtrend
            elif rsi_overbought and bear_trend_15m and ema21_slope_down:
                new_signal = -SIZE_HALF
            
            # Momentum: Lower high with trend + Supertrend
            elif lower_high and bear_trend_15m and st_bearish:
                new_signal = -SIZE_HALF
            
            # Breakdown: Price below EMA50 with momentum
            elif close[i] < ema_50[i] and rsi[i] < 50 and st_bearish:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss - trail with 2*ATR
        if position_side > 0:
            if highest_close == 0.0 or close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss - trail with 2*ATR
        if position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            # Position reversal
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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