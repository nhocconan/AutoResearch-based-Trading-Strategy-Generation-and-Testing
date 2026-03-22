#!/usr/bin/env python3
"""
Experiment #187: 15m Supertrend + 4h HMA Trend + 1h ADX Filter + ATR Stop

Hypothesis: 15m Supertrend captures short-term momentum moves while 4h HMA provides
stable higher-timeframe bias. 1h ADX > 15 filters weak/choppy markets (lower threshold
for 15m to ensure sufficient trades). ATR trailing stop at 2.0x protects against
15m noise and reversals. This should work better than the failed #175 which had
too many conflicting filters.

Why 15m might work NOW:
- 15m = 96 bars/day, captures intraday momentum without 5m noise
- Supertrend(10,3) is proven trend-following on lower timeframes
- 4h HMA(21) filter prevents counter-trend trades (proven in best strategies)
- 1h ADX > 15 (not 25) ensures enough trades on 15m timeframe
- Conservative sizing (0.25) controls drawdown
- Simple logic = fewer conflicting conditions = more trades

Learning from #175 failure (Sharpe=-4.690):
- Too many filters (4h HMA + 1h ADX + Choppiness) = no trades or wrong timing
- Choppiness index didn't help on 15m
- Need SIMPLER entry logic with strong HTF confirmation

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h HMA + 1h ADX via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing (tighter for 15m noise)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_1h_adx_atr_v1"
timeframe = "15m"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator for trend direction."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower_band[0]
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            trend[i] = 1
            supertrend[i] = max(lower_band[i], supertrend[i-1] if trend[i-1] == 1 else lower_band[i])
        else:
            trend[i] = -1
            supertrend[i] = min(upper_band[i], supertrend[i-1] if trend[i-1] == -1 else upper_band[i])
    
    return supertrend, trend

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    adx_1h = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_trend = calculate_supertrend(high, low, close, 10, 3.0)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # 1h ADX > 15 = trending market (lower threshold for 15m to ensure trades)
        trend_strength = adx_1h_aligned[i] > 15
        
        # === SUPERTREND SIGNAL ===
        # Supertrend trend = 1 (bullish) or -1 (bearish)
        st_bullish = st_trend[i] == 1
        st_bearish = st_trend[i] == -1
        
        # === EMA CONFIRMATION ===
        # EMA21 > EMA50 = bullish trend structure
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + 1h ADX trending + Supertrend bullish + EMA confirmation
        # Simpler logic = more trades (learning from #175 failure)
        if bull_trend_4h and trend_strength and st_bullish:
            if ema_bullish:
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + 1h ADX trending + Supertrend bearish + EMA confirmation
        if bear_trend_4h and trend_strength and st_bearish:
            if ema_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close (tighter for 15m)
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals