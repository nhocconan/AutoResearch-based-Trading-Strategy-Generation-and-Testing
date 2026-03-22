#!/usr/bin/env python3
"""
Experiment #039: 1h Multi-Timeframe Trend-Follow with 4h HMA Bias
Hypothesis: 1h timeframe captures short-term momentum while 4h HMA provides cleaner trend filter than 1d.
Key insight: Previous 1h strategies failed due to too many filters (CRSI, Choppiness) = 0 trades.
This strategy uses SIMPLER logic: 4h HMA for trend bias, 1h EMA pullback entries, RSI 40-60 zone, ADX>20.
Position sizing: 0.25 base, 0.15 half - discrete levels to minimize fee churn.
Timeframe: 1h (REQUIRED for exp#039), HTF: 4h via mtf_data helper.
Why this might work: 1h has more entry opportunities than 12h/1d, 4h HMA smoother than 1h for regime.
Entry conditions LOOSENED vs failed experiments to ensure 10+ trades per symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_trend_4h_hma_ema_rsi_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = pd.Series(plus_dm / (atr + 1e-10)).ewm(span=period, min_periods=period, adjust=False).mean().values * 100
    minus_di = pd.Series(minus_dm / (atr + 1e-10)).ewm(span=period, min_periods=period, adjust=False).mean().values * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    ema_200 = calculate_ema(close, 200)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # HMA on 1h for faster trend
    hma_1h = calculate_hma(close, 21)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h trend confirmation
        bull_trend_1h = ema_21[i] > ema_50[i] and close[i] > ema_50[i]
        bear_trend_1h = ema_21[i] < ema_50[i] and close[i] < ema_50[i]
        
        # ADX trend strength - LOOSENED threshold
        trend_strong = adx[i] > 18  # Lower than typical 25 for more trades
        trend_weak = adx[i] < 25
        
        # RSI conditions - LOOSENED for more trades
        rsi_pullback_long = 35 < rsi[i] < 55  # Pullback zone, not oversold
        rsi_bounce_short = 45 < rsi[i] < 65  # Bounce zone, not overbought
        rsi_momentum_long = rsi[i] > 50 and rsi[i] < 70
        rsi_momentum_short = rsi[i] < 50 and rsi[i] > 30
        
        # Price position relative to Bollinger
        price_near_lower = close[i] < bb_lower[i] * 1.02 or close[i] < bb_mid[i]
        price_near_upper = close[i] > bb_upper[i] * 0.98 or close[i] > bb_mid[i]
        
        # HMA crossover on 1h
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_1h[i]) and not np.isnan(hma_1h[i-1]):
            hma_cross_long = hma_1h[i] > ema_50[i] and hma_1h[i-1] <= ema_50[i-1]
            hma_cross_short = hma_1h[i] < ema_50[i] and hma_1h[i-1] >= ema_50[i-1]
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.015 and close[i] >= ema_21[i] * 0.985
        price_near_ema21_short = close[i] >= ema_21[i] * 0.985 and close[i] <= ema_21[i] * 1.015
        
        # DI crossover confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (when 4h bullish) ===
        if bull_trend_4h:
            # Primary: Pullback to EMA21 with RSI confirmation + ADX
            if price_near_ema21_long and rsi_pullback_long and trend_strong and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: HMA crossover with 4h confirmation + DI
            elif hma_cross_long and bull_trend_4h and di_bullish and trend_strong:
                new_signal = SIZE_BASE
            
            # Tertiary: RSI momentum in uptrend (simpler condition)
            elif rsi_momentum_long and bull_trend_1h and di_bullish:
                new_signal = SIZE_HALF
            
            # Quaternary: Price near BB mid with trend (more entries)
            elif close[i] > bb_mid[i] and bull_trend_4h and rsi[i] > 45:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (when 4h bearish) ===
        elif bear_trend_4h:
            # Primary: Bounce to EMA21 with RSI confirmation + ADX
            if price_near_ema21_short and rsi_bounce_short and trend_strong and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: HMA crossover with 4h confirmation + DI
            elif hma_cross_short and bear_trend_4h and di_bearish and trend_strong:
                new_signal = -SIZE_BASE
            
            # Tertiary: RSI momentum in downtrend (simpler condition)
            elif rsi_momentum_short and bear_trend_1h and di_bearish:
                new_signal = -SIZE_HALF
            
            # Quaternary: Price near BB mid with trend (more entries)
            elif close[i] < bb_mid[i] and bear_trend_4h and rsi[i] < 55:
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