#!/usr/bin/env python3
"""
Experiment #040: 4h Donchian Breakout with 1d HMA Trend Filter
Hypothesis: Donchian breakouts generate consistent trades across all market conditions.
1d HMA provides regime filter to only trade in trend direction (reduces whipsaws).
RSI filter avoids entering at extreme levels. ATR trailing stop manages risk.
Why this might work: Donchian is proven trend-following, generates 20-50 trades/year.
1d HMA smoother than 4h for regime detection. Simple logic = fewer failure points.
Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper.
Position sizing: 0.25 discrete, stoploss at 2.5*ATR trailing.
Must generate 10+ trades on train, 3+ on test - Donchian ensures trade frequency.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1d_hma_rsi_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    return donchian_upper, donchian_lower, donchian_mid

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

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
    
    # Donchian channels for breakout signals
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # EMA for additional trend confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_REDUCED = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - main regime filter
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # 4h trend confirmation
        bull_trend_4h = ema_21[i] > ema_50[i] if not np.isnan(ema_50[i]) else False
        bear_trend_4h = ema_21[i] < ema_50[i] if not np.isnan(ema_50[i]) else False
        
        # Donchian breakout signals
        breakout_long = high[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = low[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # RSI filter - avoid extreme entries (loosened for more trades)
        rsi_ok_long = rsi[i] < 70  # Not overbought
        rsi_ok_short = rsi[i] > 30  # Not oversold
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        # Price above/below Donchian mid for confirmation
        above_mid = close[i] > donchian_mid[i] if not np.isnan(donchian_mid[i]) else False
        below_mid = close[i] < donchian_mid[i] if not np.isnan(donchian_mid[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        if bull_trend_1d:
            # Primary: Donchian breakout with trend alignment
            if breakout_long and rsi_ok_long and rsi_momentum_long:
                new_signal = SIZE_BASE
            
            # Secondary: Price above mid + EMA confirmation
            elif above_mid and bull_trend_4h and rsi[i] > 50:
                new_signal = SIZE_REDUCED
        
        # === SHORT ENTRIES ===
        elif bear_trend_1d:
            # Primary: Donchian breakdown with trend alignment
            if breakout_short and rsi_ok_short and rsi_momentum_short:
                new_signal = -SIZE_BASE
            
            # Secondary: Price below mid + EMA confirmation
            elif below_mid and bear_trend_4h and rsi[i] < 50:
                new_signal = -SIZE_REDUCED
        
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
            # New position
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            # Position reversal
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            # Position closed
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals