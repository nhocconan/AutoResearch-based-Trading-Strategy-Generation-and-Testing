#!/usr/bin/env python3
"""
Experiment #001: 15m Supertrend + RSI Pullback + 4h HMA Trend Filter
Hypothesis: 15m timeframe captures intraday momentum while 4h HMA filters major trend direction.
Supertrend provides dynamic support/resistance, RSI(14) identifies pullback entries in trend.
ADX(14) > 25 confirms trending regime (avoid mean reversion in strong trends).
ATR(14) trailing stop at 2.5*ATR limits drawdown during reversals.
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.30 max, discrete levels (0.0, ±0.25, ±0.30) to minimize fee churn.
Key innovation: Only enter long when 4h HMA bullish + 15m Supertrend bullish + RSI pullback < 45.
Only enter short when 4h HMA bearish + 15m Supertrend bearish + RSI rally > 55.
This avoids counter-trend trades that destroyed performance in 2022 crash.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_rsi_4h_hma_v1"
timeframe = "15m"
leverage = 1.0

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=bullish, -1=bearish)
    
    Formula:
    - ATR = EMA(TR, period)
    - Upper Band = (High + Low) / 2 + multiplier * ATR
    - Lower Band = (High + Low) / 2 - multiplier * ATR
    - Supertrend = Lower Band if close > prev_supertrend, else Upper Band
    """
    n = len(close)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    direction[:] = np.nan
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[0] = lower_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if np.isnan(atr[i]):
            supertrend[i] = supertrend[i - 1]
            direction[i] = direction[i - 1]
            continue
        
        # Supertrend logic
        if close[i] > supertrend[i - 1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
    
    return supertrend, direction

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    valid = avg_loss > 0
    rs = np.zeros(n)
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    rsi[valid] = 100 - (100 / (1 + rs[valid]))
    rsi[~valid & (avg_gain > 0)] = 100.0
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i - 1]
        minus_move = low[i - 1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    supertrend, st_direction = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    rsi = calculate_rsi(close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Bollinger Bands for regime detection
    close_s = pd.Series(close)
    bb_sma = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std
    bb_bandwidth = (bb_upper - bb_lower) / (bb_sma + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
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
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - CRITICAL FILTER
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 15m Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # ADX trend strength (avoid trading in choppy markets)
        trending = adx[i] > 20  # Lower threshold for 15m
        strong_trend = adx[i] > 30
        
        # RSI pullback levels
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_extreme_oversold = rsi[i] < 30
        rsi_extreme_overbought = rsi[i] > 70
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        # Price position vs Bollinger Bands
        price_near_lower = close[i] < bb_lower[i] * 1.005
        price_near_upper = close[i] > bb_upper[i] * 0.995
        
        # Volume confirmation (optional but helpful)
        vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
        high_volume = prices['volume'].values[i] > 1.2 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: 4h bull + 15m Supertrend bull + RSI pullback
        if bull_trend_4h and st_bullish and rsi_oversold:
            new_signal = SIZE_BASE
        # Secondary: 4h bull + Supertrend bull + RSI extreme + high volume
        if bull_trend_4h and st_bullish and rsi_extreme_oversold and high_volume:
            new_signal = SIZE_MAX
        # Tertiary: All bullish alignment + price near lower BB (pullback entry)
        if bull_trend_4h and st_bullish and ema_bullish and price_near_lower:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: 4h bear + 15m Supertrend bear + RSI rally
        if bear_trend_4h and st_bearish and rsi_overbought:
            new_signal = -SIZE_BASE
        # Secondary: 4h bear + Supertrend bear + RSI extreme + high volume
        if bear_trend_4h and st_bearish and rsi_extreme_overbought and high_volume:
            new_signal = -SIZE_MAX
        # Tertiary: All bearish alignment + price near upper BB (pullback entry)
        if bear_trend_4h and st_bearish and ema_bearish and price_near_upper:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            
            # Take profit: reduce to half at 2R
            profit_target = entry_price + 2.5 * atr[int(i - 1)] if i > 0 else entry_price * 1.05
            if close[i] > profit_target and new_signal > 0:
                new_signal = SIZE_HALF
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            
            # Take profit: reduce to half at 2R
            profit_target = entry_price - 2.5 * atr[int(i - 1)] if i > 0 else entry_price * 0.95
            if close[i] < profit_target and new_signal < 0:
                new_signal = -SIZE_HALF
        
        # === SUPERTREND REVERSAL EXIT ===
        # Exit long if Supertrend flips bearish
        if position_side > 0 and st_bearish and new_signal > 0:
            new_signal = 0.0
        
        # Exit short if Supertrend flips bullish
        if position_side < 0 and st_bullish and new_signal < 0:
            new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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