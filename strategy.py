#!/usr/bin/env python3
"""
Experiment #042: 1d Daily Trend-Follow with 1w Weekly HMA Regime Filter
Hypothesis: Daily timeframe captures major trends with less noise than intraday.
Weekly HMA provides robust regime filter to avoid trading against major trend.
Key insight: Previous 1d strategies failed due to overly complex filters or too-strict conditions.
This uses simpler logic: 1w HMA for regime, 1d EMA/RSI for entries, ATR stops.
Position sizing: 0.25-0.30 discrete levels to control drawdown during 2022 crash.
Timeframe: 1d (REQUIRED for exp#042), HTF: 1w via mtf_data helper.
Why this might work: Daily bars have fewer whipsaws, weekly filter avoids bear traps.
Must generate 10+ trades on train (easy on 1d over 4 years), 3+ on test.
Entry conditions loosened vs failed experiments to ensure trade generation.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_trend_1w_hma_ema_rsi_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (high/low over period)."""
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return donchian_high, donchian_low

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = 100 * pd.Series(plus_dm / (atr + 1e-10)).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = 100 * pd.Series(minus_dm / (atr + 1e-10)).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Donchian channels for breakout detection
    donchian_high, donchian_low = calculate_donchian(high, low, 20)
    
    # ADX for trend strength
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # HMA on 1d for faster trend
    hma_1d = calculate_hma(close, 21)
    
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
        
        # ADX trend strength - LOOSENED for more trades
        trend_strong = adx[i] > 20  # Was 25, lowered to 20
        trend_weak = adx[i] < 25
        
        # RSI conditions - LOOSENED for more trades
        rsi_bullish = rsi[i] > 45 and rsi[i] < 70  # Wider range
        rsi_bearish = rsi[i] > 30 and rsi[i] < 55
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # HMA crossover on 1d
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_1d[i]) and not np.isnan(hma_1d[i-1]):
            hma_cross_long = hma_1d[i] > ema_50[i] and hma_1d[i-1] <= ema_50[i-1]
            hma_cross_short = hma_1d[i] < ema_50[i] and hma_1d[i-1] >= ema_50[i-1]
        
        # Donchian breakout
        donchian_breakout_long = close[i] > donchian_high[i-1] if not np.isnan(donchian_high[i-1]) else False
        donchian_breakout_short = close[i] < donchian_low[i-1] if not np.isnan(donchian_low[i-1]) else False
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.03 and close[i] >= ema_21[i] * 0.97
        price_near_ema21_short = close[i] >= ema_21[i] * 0.97 and close[i] <= ema_21[i] * 1.03
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 3:
            higher_low = low[i] > low[i-3]
            lower_high = high[i] < high[i-3]
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 1w bullish) ===
        if bull_trend_1w:
            # Primary: Pullback to EMA21 with RSI confirmation
            if price_near_ema21_long and rsi_bullish and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: HMA crossover with 1w confirmation
            elif hma_cross_long and bull_trend_1w and di_bullish:
                new_signal = SIZE_BASE
            
            # Tertiary: RSI oversold bounce in uptrend
            elif rsi_oversold and bull_trend_1d and trend_strong:
                new_signal = SIZE_HALF
            
            # Breakout: Donchian breakout with trend
            elif donchian_breakout_long and bull_trend_1d and rsi[i] > 50:
                new_signal = SIZE_BASE
            
            # Momentum: Higher low with trend
            elif higher_low and bull_trend_1d and rsi[i] > 45:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 1w bearish) ===
        elif bear_trend_1w:
            # Primary: Bounce to EMA21 with RSI confirmation
            if price_near_ema21_short and rsi_bearish and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: HMA crossover with 1w confirmation
            elif hma_cross_short and bear_trend_1w and di_bearish:
                new_signal = -SIZE_BASE
            
            # Tertiary: RSI overbought rejection in downtrend
            elif rsi_overbought and bear_trend_1d and trend_strong:
                new_signal = -SIZE_HALF
            
            # Breakdown: Donchian breakdown with trend
            elif donchian_breakout_short and bear_trend_1d and rsi[i] < 50:
                new_signal = -SIZE_BASE
            
            # Momentum: Lower high with trend
            elif lower_high and bear_trend_1d and rsi[i] < 55:
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