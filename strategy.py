#!/usr/bin/env python3
"""
Experiment #158: 30m EMA Crossover + 4h HMA Trend Filter + ATR Stop

Hypothesis: 30m timeframe offers balance between noise (15m) and lag (1h).
Simple EMA crossover (8/21) provides clear entry signals while 4h HMA 
filters counter-trend trades. This is simpler than failed experiments 
(#151-#157) which had too many filters causing negative Sharpe.

Why this might work:
- 30m has not been successfully tested yet (required for this experiment)
- EMA 8/21 crossover is proven in literature for medium-term trends
- 4h HMA provides stable trend bias (used in best strategies)
- Simple conditions ensure adequate trade frequency (avoid 0-trade problem)
- ATR stoploss protects from 2022-style crashes

Learning from failures:
- #151-#157: All negative Sharpe, too many filters (CHOP, CRSI, RSI, etc.)
- #142, #143: 0 trades from overly strict conditions
- Keep it SIMPLE: trend filter + entry signal + stoploss

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_ema_4h_hma_trend_atr_v1"
timeframe = "30m"
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

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === EMA CROSSOVER SIGNAL ===
        # Fast EMA crosses above slow EMA = bullish
        # Fast EMA crosses below slow EMA = bearish
        ema_bull = ema_fast[i] > ema_slow[i]
        ema_bear = ema_fast[i] < ema_slow[i]
        
        # EMA crossover detection (cross happened this bar)
        ema_cross_bull = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_bear = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # === VOLUME CONFIRMATION ===
        # Volume above average confirms the move
        vol_confirmed = volume[i] > vol_ma[i] * 1.0
        
        # === PRICE VS 50 EMA FILTER ===
        # Avoid entries against medium-term trend
        price_above_50 = close[i] > ema_50[i]
        price_below_50 = close[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # 4h bullish + EMA bullish + price above 50 EMA
        # Entry on crossover OR already in bullish state
        if bull_trend_4h and ema_bull and price_above_50:
            if ema_cross_bull or (in_position and position_side > 0):
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # 4h bearish + EMA bearish + price below 50 EMA
        if bear_trend_4h and ema_bear and price_below_50:
            if ema_cross_bear or (in_position and position_side < 0):
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals