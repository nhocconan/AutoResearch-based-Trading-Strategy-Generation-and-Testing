#!/usr/bin/env python3
"""
Experiment #002: 30m Volatility Breakout with 4h HMA Regime Filter
Hypothesis: Volatility expansion breakouts work better than mean reversion in crypto.
Use 4h HMA to determine bull/bear regime, then trade 30m Donchian breakouts only in trend direction.
RSI filter avoids chasing overextended moves. ATR expansion confirms genuine breakout vs fakeout.
This differs from #001 (CRSI+Chop mean reversion) and #008 (HMA pullback entries).
Key insight: Crypto trends persist once volatility expands. Catch the expansion early with HTF filter.
Position sizing: 0.25 base, 0.15 half, stoploss at 2.5*ATR trailing.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_vol_breakout_4h_hma_v1"
timeframe = "30m"
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
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_bb(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_volatility_ratio(close, short_period=7, long_period=30):
    """Calculate volatility expansion ratio (short-term vol / long-term vol)."""
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0)
    
    short_vol = pd.Series(returns).rolling(window=short_period, min_periods=short_period).std().values
    long_vol = pd.Series(returns).rolling(window=long_period, min_periods=long_period).std().values
    
    ratio = np.zeros(len(close))
    mask = long_vol > 0
    ratio[mask] = short_vol[mask] / long_vol[mask]
    ratio[~mask] = 1.0
    
    return ratio

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    
    # Donchian Channel for breakouts
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Bollinger Bands for volatility context
    bb_upper, bb_lower, bb_mid = calculate_bb(close, 20, 2.0)
    
    # Volatility expansion ratio
    vol_ratio = calculate_volatility_ratio(close, 7, 30)
    
    # HMA on 30m for faster trend confirmation
    hma_30m = calculate_hma(close, 21)
    
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend regime (HTF) - main directional filter
        bull_regime_4h = close[i] > hma_4h_aligned[i]
        bear_regime_4h = close[i] < hma_4h_aligned[i]
        
        # 30m trend confirmation
        bull_trend_30m = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_30m = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = close[i] > ema_200[i] if not np.isnan(ema_200[i]) else True
        below_200 = close[i] < ema_200[i] if not np.isnan(ema_200[i]) else False
        
        # Volatility expansion confirmation (breakout needs vol expansion)
        vol_expanding = vol_ratio[i] > 1.3  # Short-term vol 30% higher than long-term
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if i >= 1 else False
        breakout_short = close[i] < donchian_lower[i - 1] if i >= 1 else False
        
        # Price near Donchian (about to breakout)
        near_breakout_long = close[i] > donchian_upper[i] * 0.98 if not np.isnan(donchian_upper[i]) else False
        near_breakout_short = close[i] < donchian_lower[i] * 1.02 if not np.isnan(donchian_lower[i]) else False
        
        # RSI filter - avoid overextended entries
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        rsi_momentum_long = rsi[i] > 50 and rsi[i] > rsi[i - 2] if i >= 2 else False
        rsi_momentum_short = rsi[i] < 50 and rsi[i] < rsi[i - 2] if i >= 2 else False
        
        # ATR expansion confirmation
        atr_expanding = atr[i] > atr[i - 5] * 1.1 if i >= 5 else False
        
        # Bollinger Band width context (squeeze = potential explosion)
        bb_width = (bb_upper[i] - bb_lower[i]) / bb_mid[i] if not np.isnan(bb_mid[i]) else 0
        bb_squeeze = bb_width < np.nanmean(bb_width[max(0, i-50):i]) * 0.7 if i >= 50 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 4h bullish regime) ===
        if bull_regime_4h:
            # Primary: Donchian breakout with vol expansion
            if breakout_long and vol_expanding and rsi_not_overbought and above_200:
                new_signal = SIZE_BASE
            
            # Secondary: Near breakout with RSI momentum
            elif near_breakout_long and rsi_momentum_long and bull_trend_30m:
                new_signal = SIZE_HALF
            
            # Tertiary: BB squeeze breakout with ATR expansion
            elif bb_squeeze and breakout_long and atr_expanding and rsi_not_overbought:
                new_signal = SIZE_BASE
            
            # Continuation: HMA crossover with 4h confirmation
            elif hma_30m[i] > ema_50[i] and hma_30m[i - 1] <= ema_50[i - 1] if i >= 1 else False:
                if bull_regime_4h and rsi_momentum_long:
                    new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 4h bearish regime) ===
        elif bear_regime_4h:
            # Primary: Donchian breakdown with vol expansion
            if breakout_short and vol_expanding and rsi_not_oversold and below_200:
                new_signal = -SIZE_BASE
            
            # Secondary: Near breakdown with RSI momentum
            elif near_breakout_short and rsi_momentum_short and bear_trend_30m:
                new_signal = -SIZE_HALF
            
            # Tertiary: BB squeeze breakdown with ATR expansion
            elif bb_squeeze and breakout_short and atr_expanding and rsi_not_oversold:
                new_signal = -SIZE_BASE
            
            # Continuation: HMA crossover with 4h confirmation
            elif hma_30m[i] < ema_50[i] and hma_30m[i - 1] >= ema_50[i - 1] if i >= 1 else False:
                if bear_regime_4h and rsi_momentum_short:
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