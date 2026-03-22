#!/usr/bin/env python3
"""
Experiment #090: 1d Supertrend + 1w HMA Trend Filter + ADX Regime + RSI Momentum
Hypothesis: Daily timeframe captures major trend moves with minimal noise.
Supertrend (ATR-based) is proven on longer timeframes (#083, #088 showed positive Sharpe).
1w HMA provides ultra-stable trend bias (slower than 1d, avoids whipsaws).
ADX filter ensures we only trade in trending markets (ADX>20).
RSI momentum confirms entries without being too restrictive.

Why this might work on 1d:
- #088 (4h Supertrend + 1d HMA + ADX): Sharpe=0.223 - Supertrend + HTF works!
- #083 (12h Supertrend + 1d HMA + RSI): Sharpe=0.085 - Supertrend on longer TF works
- #084 (1d Supertrend + 4h HMA): Sharpe=-0.110 - Wrong HTF (4h too fast for 1d)
- Key insight: 1d needs 1w HTF (not 4h) for proper trend hierarchy
- Simplified entry conditions to ensure enough trades on all symbols

Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong signals. Stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_1w_hma_adx_rsi_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = long (price above supertrend), -1 = short
    
    for i in range(1, n):
        if np.isnan(atr[i]):
            supertrend[i] = np.nan
            direction[i] = 0
            continue
        
        # Update bands based on previous direction
        if direction[i-1] == 1:  # Previous was long
            lower_band[i] = max(lower_band[i], supertrend[i-1])
            if close[i] < lower_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
        else:  # Previous was short
            upper_band[i] = min(upper_band[i], supertrend[i-1])
            if close[i] > upper_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
    
    supertrend[0] = lower_band[0]
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = np.zeros(n)
    mask = (plus_di + minus_di) > 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias (ultra-stable)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === SUPERTREND SIGNAL ===
        # st_direction = 1 means price above supertrend (bullish)
        # st_direction = -1 means price below supertrend (bearish)
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ADX REGIME FILTER ===
        # ADX > 20 = trending market (good for trend following)
        # ADX < 15 = ranging market (exit positions)
        trending_market = adx[i] > 20
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 15
        
        # === RSI MOMENTUM ===
        rsi_momentum_long = rsi[i] > 45  # Not deeply oversold
        rsi_momentum_short = rsi[i] < 55  # Not deeply overbought
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths to ensure trades) ===
        # Path 1: Supertrend bullish + 1w bullish + strong trend (primary - strong signal)
        if st_bullish and bull_trend_1w and strong_trend:
            if ema_bullish:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # Path 2: Supertrend bullish + EMA bullish + trending (simpler, ensures trades)
        if new_signal == 0.0 and st_bullish and ema_bullish and trending_market:
            if bull_trend_1w or rsi_momentum_long:
                new_signal = SIZE_BASE
        
        # Path 3: Supertrend bullish + 1w bullish only (fallback to ensure trades)
        if new_signal == 0.0 and st_bullish and bull_trend_1w:
            if trending_market or ema_bullish:
                new_signal = SIZE_BASE
        
        # Path 4: Supertrend bullish + trending market (minimal filter for trade generation)
        if new_signal == 0.0 and st_bullish and trending_market:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths to ensure trades) ===
        # Path 1: Supertrend bearish + 1w bearish + strong trend (primary - strong signal)
        if st_bearish and bear_trend_1w and strong_trend:
            if ema_bearish:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # Path 2: Supertrend bearish + EMA bearish + trending (simpler, ensures trades)
        if new_signal == 0.0 and st_bearish and ema_bearish and trending_market:
            if bear_trend_1w or rsi_momentum_short:
                new_signal = -SIZE_BASE
        
        # Path 3: Supertrend bearish + 1w bearish only (fallback to ensure trades)
        if new_signal == 0.0 and st_bearish and bear_trend_1w:
            if trending_market or ema_bearish:
                new_signal = -SIZE_BASE
        
        # Path 4: Supertrend bearish + trending market (minimal filter for trade generation)
        if new_signal == 0.0 and st_bearish and trending_market:
            new_signal = -SIZE_BASE
        
        # === EXIT CONDITIONS ===
        # Exit if ADX drops too low (trend dying)
        if in_position and weak_trend:
            new_signal = 0.0
        
        # Exit if Supertrend reverses against position
        if in_position and position_side > 0 and st_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and st_bullish:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR for 1d ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            # Trailing stop: 2.5 * ATR below highest since entry
            stoploss_price = highest_since_entry - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0 or close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            # Trailing stop: 2.5 * ATR above lowest since entry
            stoploss_price = lowest_since_entry + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking for next iteration
        if new_signal != 0.0:
            if not in_position or np.sign(new_signal) != position_side:
                # New position or reversal
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif in_position:
                # Same position, update extremes
                if position_side > 0 and close[i] > highest_since_entry:
                    highest_since_entry = close[i]
                if position_side < 0 and (lowest_since_entry == 0.0 or close[i] < lowest_since_entry):
                    lowest_since_entry = close[i]
        else:
            # Exiting position
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals