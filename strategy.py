#!/usr/bin/env python3
"""
Experiment #014: 30m CRSI Mean Reversion + 4h HMA Trend + Volume Filter
Hypothesis: Connors RSI (CRSI) at 30m timeframe captures more frequent mean reversion 
opportunities than 1h. Combined with 4h HMA trend filter to avoid counter-trend trades.
Volume spike confirmation reduces false signals. ATR trailing stop limits drawdown.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25 base, 0.35 max, discrete levels to minimize fee churn.
Key innovation: Looser CRSI thresholds (<20/>80 vs <10/>90) to generate MORE trades.
Volume confirmation: entry volume > 1.5x 20-bar avg volume.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_4h_hma_volume_v1"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines 3 components for mean reversion signals.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    if n < rank_period + 10:
        return crsi
    
    # Component 1: RSI(3) - vectorized
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi3 = 100 - (100 / (1 + rs))
    rsi3 = rsi3.values
    
    # Component 2: Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1
        else:
            streak[i] = streak[i - 1]
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = np.abs(streak[max(0, i - streak_period + 1):i + 1])
        if len(streak_vals) > 0:
            streak_rsi[i] = np.mean(streak_vals) / streak_period * 100
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: Percent Rank(100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        percent_rank[i] = rank / (rank_period - 1) * 100
    
    # Combine components
    valid_mask = (~np.isnan(rsi3)) & (~np.isnan(streak_rsi)) & (~np.isnan(percent_rank))
    crsi[valid_mask] = (rsi3[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, bandwidth, sma

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Volume moving average for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Additional trend filters
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
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
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # CRSI signals - LOOSENED thresholds for MORE trades
        crsi_oversold = crsi[i] < 25  # Was <15, now <25 for more signals
        crsi_overbought = crsi[i] > 75  # Was >85, now >75 for more signals
        crsi_extreme_oversold = crsi[i] < 15
        crsi_extreme_overbought = crsi[i] > 85
        
        # Price position vs Bollinger Bands
        price_near_lower = close[i] < bb_lower[i] * 1.01  # Within 1% of lower band
        price_near_upper = close[i] > bb_upper[i] * 0.99  # Within 1% of upper band
        price_below_lower = close[i] < bb_lower[i]
        price_above_upper = close[i] > bb_upper[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.3 * vol_sma[i]  # 30% above avg
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_21[i] and close[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and close[i] < ema_50[i]
        
        # Overall market trend
        market_bull = close[i] > ema_200[i]
        market_bear = close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY === (multiple conditions to trigger MORE often)
        # Primary: CRSI extreme oversold + 4h bull trend (no BB requirement)
        if crsi_extreme_oversold and bull_trend:
            new_signal = SIZE_MAX
        # Secondary: CRSI oversold + price near lower BB + 4h bull trend
        elif crsi_oversold and price_near_lower and bull_trend:
            new_signal = SIZE_BASE
        # Tertiary: CRSI oversold + volume spike + 4h bull trend
        elif crsi_oversold and volume_spike and bull_trend:
            new_signal = SIZE_BASE
        # Quaternary: CRSI oversold + price below lower BB + market bull
        elif crsi_oversold and price_below_lower and market_bull:
            new_signal = SIZE_BASE
        # Fifth: CRSI oversold + EMA bullish + 4h bull trend
        elif crsi_oversold and ema_bullish and bull_trend:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY === (multiple conditions to trigger MORE often)
        # Primary: CRSI extreme overbought + 4h bear trend (no BB requirement)
        if crsi_extreme_overbought and bear_trend:
            new_signal = -SIZE_MAX
        # Secondary: CRSI overbought + price near upper BB + 4h bear trend
        elif crsi_overbought and price_near_upper and bear_trend:
            new_signal = -SIZE_BASE
        # Tertiary: CRSI overbought + volume spike + 4h bear trend
        elif crsi_overbought and volume_spike and bear_trend:
            new_signal = -SIZE_BASE
        # Quaternary: CRSI overbought + price above upper BB + market bear
        elif crsi_overbought and price_above_upper and market_bear:
            new_signal = -SIZE_BASE
        # Fifth: CRSI overbought + EMA bearish + 4h bear trend
        elif crsi_overbought and ema_bearish and bear_trend:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals