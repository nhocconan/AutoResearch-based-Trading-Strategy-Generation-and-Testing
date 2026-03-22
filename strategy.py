#!/usr/bin/env python3
"""
Experiment #010: 4h Multi-Timeframe Trend-Follow with 1d HMA Bias

Hypothesis: 4h timeframe balances signal frequency and noise reduction.
1d HMA provides regime filter (bull/bear), 4h EMA crossover for entries,
RSI for pullback confirmation, ATR for stops.

Why this might work:
- 12h strategy (#005) had Sharpe=-0.39, closest to positive among all experiments
- 4h should generate MORE trades than 12h (addressing trade count concerns)
- Trend-following works better than mean-reversion on crypto (learned from failures)
- 1d HMA smoother than 4h for regime detection (proven in #005)
- Loosened entry conditions vs failed vol breakout (0 trades)

Key changes from #005:
- Primary TF: 4h instead of 12h (more trades)
- HTF: 1d HMA (same as #005, worked well)
- Entry: EMA21/50 crossover + RSI pullback (simpler than #005)
- Stop: 2.5*ATR trailing (same as #005)
- Position size: 0.25-0.30 discrete (same as #005)

Must generate 10+ trades on train, 3+ on test.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trend_1d_hma_ema_rsi_v1"
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
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = pd.Series(plus_dm / (atr + 1e-10)).ewm(span=period, min_periods=period, adjust=False).mean().values * 100
    minus_di = pd.Series(minus_dm / (atr + 1e-10)).ewm(span=period, min_periods=period, adjust=False).mean().values * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 4h trend confirmation
        bull_trend_4h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_4h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # ADX trend strength filter
        adx_strong = adx[i] > 20  # Trending market
        adx_weak = adx[i] < 25   # Range market
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # RSI conditions - LOOSENED for more trades (learned from 0-trade failures)
        rsi_pullback_long = 35 < rsi[i] < 65  # Wider range
        rsi_bounce_short = 35 < rsi[i] < 65
        rsi_oversold = rsi[i] < 45
        rsi_overbought = rsi[i] > 55
        
        # HMA crossover on 4h - primary entry signal
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_4h[i]) and not np.isnan(hma_4h[i-1]):
            hma_cross_long = hma_4h[i] > ema_50[i] and hma_4h[i-1] <= ema_50[i-1]
            hma_cross_short = hma_4h[i] < ema_50[i] and hma_4h[i-1] >= ema_50[i-1]
        
        # EMA crossover - secondary entry
        ema_cross_long = False
        ema_cross_short = False
        if i >= 1:
            ema_cross_long = ema_21[i] > ema_50[i] and ema_21[i-1] <= ema_50[i-1]
            ema_cross_short = ema_21[i] < ema_50[i] and ema_21[i-1] >= ema_50[i-1]
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.03 and close[i] >= ema_21[i] * 0.97
        price_near_ema21_short = close[i] >= ema_21[i] * 0.97 and close[i] <= ema_21[i] * 1.03
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 3:
            higher_low = low[i] > low[i-3]
            lower_high = high[i] < high[i-3]
        
        # DI crossover for momentum
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 1d bullish) ===
        if bull_trend_1d:
            # Primary: HMA crossover with 1d confirmation
            if hma_cross_long and bull_trend_1d and di_bullish:
                new_signal = SIZE_BASE
            
            # Secondary: EMA21/50 crossover with RSI confirmation
            elif ema_cross_long and bull_trend_1d and rsi_pullback_long:
                new_signal = SIZE_BASE
            
            # Tertiary: Pullback to EMA21 in uptrend
            elif price_near_ema21_long and bull_trend_4h and rsi[i] > 40:
                new_signal = SIZE_HALF
            
            # Momentum: Higher low with trend + ADX
            elif higher_low and bull_trend_4h and adx_strong and rsi[i] > 45:
                new_signal = SIZE_HALF
            
            # RSI oversold bounce in strong uptrend
            elif rsi_oversold and bull_trend_1d and above_200:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 1d bearish) ===
        elif bear_trend_1d:
            # Primary: HMA crossover with 1d confirmation
            if hma_cross_short and bear_trend_1d and di_bearish:
                new_signal = -SIZE_BASE
            
            # Secondary: EMA21/50 crossover with RSI confirmation
            elif ema_cross_short and bear_trend_1d and rsi_bounce_short:
                new_signal = -SIZE_BASE
            
            # Tertiary: Bounce to EMA21 in downtrend
            elif price_near_ema21_short and bear_trend_4h and rsi[i] < 60:
                new_signal = -SIZE_HALF
            
            # Momentum: Lower high with trend + ADX
            elif lower_high and bear_trend_4h and adx_strong and rsi[i] < 55:
                new_signal = -SIZE_HALF
            
            # RSI overbought rejection in strong downtrend
            elif rsi_overbought and bear_trend_1d and below_200:
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