#!/usr/bin/env python3
"""
Experiment #080: 30m EMA Momentum with 4h HMA Trend Filter + Volume Confirmation
Hypothesis: 30m timeframe captures momentum moves without excessive noise. Previous 30m strategies
failed due to too many conflicting filters (CRSI + ADX + multiple TFs). This strategy simplifies:
- 4h HMA(21) for primary trend bias (long only when price > 4h HMA, short when <)
- 30m EMA(9)/EMA(21) crossover for entry timing
- Volume confirmation via taker_buy_volume ratio (avoid low-volume false breakouts)
- ATR(14) trailing stop at 3.0x for wider stops (30m has more noise than 4h)
- Simple discrete position sizing (0.25 base, 0.30 strong)

Key insight from failures: Strategies with 4+ filters generate 0 trades or get stopped immediately.
This uses only 3 filters: HTF trend + LTF momentum + volume. Fewer conflicts = more trades.
30m is fast enough for momentum but slow enough to avoid 5m/15m whipsaw.

Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_ema_momentum_4h_hma_vol_v1"
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
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_momentum(close, period=10):
    """Calculate rate of change momentum."""
    return (close - np.roll(close, period)) / np.roll(close, period) * 100

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    ema_9 = calculate_ema(close, 9)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    momentum = calculate_momentum(close, 10)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.zeros(n)
    mask = volume > 0
    vol_ratio[mask] = taker_buy_vol[mask] / volume[mask]
    vol_ratio[~mask] = 0.5
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = primary trend direction
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 30m EMA CROSSOVER SIGNALS ===
        # EMA cross above (bullish)
        ema_cross_long = ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]
        # EMA cross below (bearish)
        ema_cross_short = ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]
        
        # EMA alignment (sustained trend)
        ema_bullish = ema_9[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = ema_9[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # === VOLUME CONFIRMATION ===
        # Volume above 20-bar MA
        vol_confirmed = volume[i] > vol_ma[i] * 0.8
        
        # Taker buy ratio confirmation
        vol_buy_pressure = vol_ratio[i] > 0.52  # More buying than selling
        vol_sell_pressure = vol_ratio[i] < 0.48  # More selling than buying
        
        # === MOMENTUM CONFIRMATION ===
        mom_positive = momentum[i] > 0.5
        mom_negative = momentum[i] < -0.5
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        
        # Path 1: EMA crossover + 4h trend bullish + volume confirmed
        if ema_cross_long and bull_trend_4h:
            if vol_confirmed and vol_buy_pressure:
                new_signal = SIZE_STRONG
            elif vol_confirmed or mom_positive:
                new_signal = SIZE_BASE
        
        # Path 2: EMA alignment + 4h trend + momentum (trend continuation)
        if bull_trend_4h and ema_bullish:
            if mom_positive and vol_ratio[i] > 0.50:
                new_signal = SIZE_BASE
        
        # Path 3: Simple trend continuation (ensure trades happen - loosened)
        if bull_trend_4h:
            if ema_9[i] > ema_21[i] and close[i] > ema_21[i]:
                if vol_ratio[i] > 0.48:  # Very loose volume filter
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        
        # Path 1: EMA crossover + 4h trend bearish + volume confirmed
        if ema_cross_short and bear_trend_4h:
            if vol_confirmed and vol_sell_pressure:
                new_signal = -SIZE_STRONG
            elif vol_confirmed or mom_negative:
                new_signal = -SIZE_BASE
        
        # Path 2: EMA alignment + 4h trend + momentum (trend continuation)
        if bear_trend_4h and ema_bearish:
            if mom_negative and vol_ratio[i] < 0.50:
                new_signal = -SIZE_BASE
        
        # Path 3: Simple trend continuation (ensure trades happen - loosened)
        if bear_trend_4h:
            if ema_9[i] < ema_21[i] and close[i] < ema_21[i]:
                if vol_ratio[i] < 0.52:  # Very loose volume filter
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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