#!/usr/bin/env python3
"""
Experiment #003: 1h Multi-Timeframe Asymmetric Trend with 4h HMA Regime Filter
Hypothesis: 1h timeframe provides better signal-to-noise than 30m for BTC/ETH.
4h HMA determines bull/bear regime. 1h RSI + EMA pullback for entries.
Asymmetric logic: Long only when 4h HMA bullish + 1h pullback. Short only when 4h HMA bearish + 1h bounce.
Key improvements over failed #001/#002:
- Simpler entry conditions (RSI range not too narrow)
- 1h TF reduces noise vs 15m/30m
- ATR trailing stop at 2.5*ATR
- Discrete sizing (0.30 base, 0.15 half) to minimize fee churn
- Must generate trades on ALL symbols (BTC/ETH/SOL) - entry conditions loosened
Timeframe: 1h (REQUIRED for exp#003), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_asymmetric_4h_hma_rsi_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = atr > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / atr[mask]
    
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    sma_200 = calculate_sma(close, 200)
    adx = calculate_adx(high, low, close, 14)
    
    # Bollinger Bands for mean reversion
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # HMA on 1h for faster trend
    hma_1h = calculate_hma(close, 21)
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h trend confirmation
        bull_trend_1h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_1h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # ADX trend strength
        trend_strong = adx[i] > 20 if not np.isnan(adx[i]) else False
        trend_weak = adx[i] < 20 if not np.isnan(adx[i]) else True
        
        # RSI conditions - LOOSENED for more trades
        rsi_pullback_long = 30 < rsi[i] < 60  # Wider range for entries
        rsi_bounce_short = 40 < rsi[i] < 70   # Wider range for entries
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Bollinger Band conditions
        price_near_bb_lower = close[i] < bb_lower[i] * 1.01
        price_near_bb_upper = close[i] > bb_upper[i] * 0.99
        price_near_bb_mid = abs(close[i] - bb_mid[i]) < atr[i] * 0.5
        
        # HMA crossover on 1h
        hma_cross_long = hma_1h[i] > ema_50[i] and hma_1h[i-1] <= ema_50[i-1] if i >= 1 else False
        hma_cross_short = hma_1h[i] < ema_50[i] and hma_1h[i-1] >= ema_50[i-1] if i >= 1 else False
        
        # Price action: higher low for long, lower high for short
        higher_low = low[i] > low[i-5] * 0.995 if i >= 5 else False
        lower_high = high[i] < high[i-5] * 1.005 if i >= 5 else False
        
        # EMA pullback entries
        pullback_to_ema21_long = close[i] <= ema_21[i] * 1.02 and close[i] >= ema_21[i] * 0.98
        bounce_to_ema21_short = close[i] >= ema_21[i] * 0.98 and close[i] <= ema_21[i] * 1.02
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 4h bullish) ===
        if bull_trend_4h:
            # Primary: Pullback to EMA21 with RSI confirmation (most common)
            if pullback_to_ema21_long and rsi_pullback_long and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: Bollinger lower touch with RSI oversold
            elif price_near_bb_lower and rsi_oversold:
                new_signal = SIZE_BASE
            
            # Tertiary: HMA crossover with trend confirmation
            elif hma_cross_long and bull_trend_1h:
                new_signal = SIZE_HALF
            
            # Momentum: RSI rising from oversold in uptrend
            elif rsi_oversold and i >= 2 and rsi[i] > rsi[i-2] and bull_trend_1h:
                new_signal = SIZE_HALF
            
            # Breakout: Price above EMA50 with strong ADX
            elif close[i] > ema_50[i] and trend_strong and above_200:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 4h bearish) ===
        elif bear_trend_4h:
            # Primary: Bounce to EMA21 with RSI confirmation (most common)
            if bounce_to_ema21_short and rsi_bounce_short and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: Bollinger upper touch with RSI overbought
            elif price_near_bb_upper and rsi_overbought:
                new_signal = -SIZE_BASE
            
            # Tertiary: HMA crossover with trend confirmation
            elif hma_cross_short and bear_trend_1h:
                new_signal = -SIZE_HALF
            
            # Momentum: RSI falling from overbought in downtrend
            elif rsi_overbought and i >= 2 and rsi[i] < rsi[i-2] and bear_trend_1h:
                new_signal = -SIZE_HALF
            
            # Breakdown: Price below EMA50 with strong ADX
            elif close[i] < ema_50[i] and trend_strong and below_200:
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