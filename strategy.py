#!/usr/bin/env python3
"""
Experiment #006: 1d Donchian Breakout with 1w HMA Trend Filter
Hypothesis: Daily timeframe reduces noise and whipsaws compared to intraday.
Weekly HMA provides strong regime filter to avoid counter-trend trades.
Donchian breakout (20-period) captures sustained moves. RSI confirms momentum.
Position sizing: 0.25-0.35 discrete levels with ATR trailing stop.
Why this might work: 1d has fewer false signals, 1w HMA smoother than 1d for regime,
Donchian breakouts work well in trending markets (2021 bull, 2022 bear).
Must generate 10+ trades on train - entry conditions loosened vs pure breakout.
Timeframe: 1d (REQUIRED for exp#006), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_v1"
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
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    
    plus_dm = np.where((high - high_shift) > (low_shift - low), 
                       np.maximum(high - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low) > (high - high_shift),
                        np.maximum(low_shift - low, 0), 0)
    
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
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
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Donchian channels
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # EMA for trend confirmation
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # SMA for long-term trend
    sma_200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
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
        bull_trend_1d = ema_21[i] > ema_50[i] and close[i] > ema_50[i]
        bear_trend_1d = ema_21[i] < ema_50[i] and close[i] < ema_50[i]
        
        # ADX trend strength
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 20
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # RSI conditions - LOOSENED for more trades
        rsi_bullish = rsi[i] > 45 and rsi[i] < 70
        rsi_bearish = rsi[i] > 30 and rsi[i] < 55
        rsi_neutral = 35 < rsi[i] < 65
        
        # Donchian breakout signals
        breakout_long = close[i] > donch_upper[i-1] if not np.isnan(donch_upper[i-1]) else False
        breakout_short = close[i] < donch_lower[i-1] if not np.isnan(donch_lower[i-1]) else False
        
        # Price near Donchian bands (for pullback entries)
        near_upper = close[i] >= donch_upper[i] * 0.98 if not np.isnan(donch_upper[i]) else False
        near_lower = close[i] <= donch_lower[i] * 1.02 if not np.isnan(donch_lower[i]) else False
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 1w bullish) ===
        if bull_trend_1w:
            # Primary: Donchian breakout with trend confirmation
            if breakout_long and bull_trend_1d and trend_strong:
                new_signal = SIZE_BASE
            
            # Secondary: Pullback to lower band in uptrend
            elif near_lower and bull_trend_1d and rsi_bullish:
                new_signal = SIZE_HALF
            
            # Tertiary: DI bullish + RSI confirmation
            elif di_bullish and bull_trend_1d and rsi[i] > 50:
                new_signal = SIZE_HALF
            
            # Momentum: Above 200 SMA with RSI
            elif above_200 and rsi_bullish and close[i] > ema_21[i]:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 1w bearish) ===
        elif bear_trend_1w:
            # Primary: Donchian breakout with trend confirmation
            if breakout_short and bear_trend_1d and trend_strong:
                new_signal = -SIZE_BASE
            
            # Secondary: Bounce to upper band in downtrend
            elif near_upper and bear_trend_1d and rsi_bearish:
                new_signal = -SIZE_HALF
            
            # Tertiary: DI bearish + RSI confirmation
            elif di_bearish and bear_trend_1d and rsi[i] < 50:
                new_signal = -SIZE_HALF
            
            # Momentum: Below 200 SMA with RSI
            elif below_200 and rsi_bearish and close[i] < ema_21[i]:
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