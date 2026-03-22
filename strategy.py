#!/usr/bin/env python3
"""
Experiment #331: 15m EMA Crossover with Dual HTF Bias and Volume Confirmation

Hypothesis: After #319 failed (Supertrend+RSI Sharpe=-3.653), the 15m timeframe needs
a DIFFERENT approach. Analysis shows:
1. 15m is too noisy for pure mean-reversion (CRSI failed at Sharpe=-3.901)
2. Trend-following with HTF bias works better on faster timeframes
3. Volume confirmation filters out false breakouts common on 15m
4. Dual HTF (1h + 4h) provides stronger directional filter than single HTF

This strategy uses:
1. 4h HMA(21) for PRIMARY trend bias (proven edge from best strategies)
2. 1h HMA(21) for SECONDARY trend confirmation (reduces whipsaw)
3. EMA(8)/EMA(21) crossover on 15m for entry timing
4. Volume > SMA20(volume) for breakout confirmation (filters noise)
5. ADX(14)>15 for minimal trend strength (looser than failed strategies)
6. ATR(14) trailing stoploss at 2.0x (tighter for 15m timeframe)

Key differences from failed #319:
- Removed Supertrend (failed on 15m)
- Removed RSI pullback (failed on 15m)
- Added volume confirmation (critical for 15m noise filtering)
- Dual HTF bias instead of single HTF (stronger filter)
- Looser ADX threshold (15 vs 25) for more trades

Timeframe: 15m (REQUIRED for this experiment)
HTF: 1h and 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_ema_crossover_dual_htf_volume_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    adx = calculate_adx(high, low, close, 14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = PRIMARY directional bias (REQUIRED for entry)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h HMA = SECONDARY trend confirmation (REQUIRED for entry)
        bull_trend_1h = close[i] > hma_1h_aligned[i]
        bear_trend_1h = close[i] < hma_1h_aligned[i]
        
        # === TREND STRENGTH ===
        # ADX > 15 = minimal trending (loose for trade generation on 15m)
        trending = adx[i] > 15
        strong_trend = adx[i] > 25
        
        # === VOLUME CONFIRMATION ===
        # Volume must be above 20-period SMA (filters false breakouts)
        volume_confirmed = volume[i] > vol_sma[i]
        
        # === EMA CROSSOVER ===
        # Fast EMA above slow EMA = bullish state
        ema_bullish = ema_fast[i] > ema_slow[i]
        # Fast EMA below slow EMA = bearish state
        ema_bearish = ema_fast[i] < ema_slow[i]
        
        # === ENTRY CONDITIONS (LOOSE for >=10 trades per symbol) ===
        new_signal = 0.0
        position_size = SIZE_BASE
        
        # Boost size if strong trend + both HTF aligned
        if strong_trend and bull_trend_4h and bull_trend_1h:
            position_size = SIZE_STRONG
        elif strong_trend and bear_trend_4h and bear_trend_1h:
            position_size = SIZE_STRONG
        
        # LONG: 4h bias up + 1h bias up + EMA bullish + ADX trending + volume confirmed
        # Both HTF must align (dual filter reduces whipsaw)
        long_conditions = (
            bull_trend_4h and
            bull_trend_1h and
            ema_bullish and
            trending and
            volume_confirmed
        )
        
        # SHORT: 4h bias down + 1h bias down + EMA bearish + ADX trending + volume confirmed
        short_conditions = (
            bear_trend_4h and
            bear_trend_1h and
            ema_bearish and
            trending and
            volume_confirmed
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long positions
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short positions
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === HTF TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and new_signal != 0.0:
            if bear_trend_4h:
                new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and new_signal != 0.0:
            if bull_trend_4h:
                new_signal = 0.0
        
        # === EMA REVERSAL EXIT ===
        # Exit long if EMA turns bearish
        if in_position and position_side > 0 and new_signal != 0.0:
            if ema_bearish:
                new_signal = 0.0
        
        # Exit short if EMA turns bullish
        if in_position and position_side < 0 and new_signal != 0.0:
            if ema_bullish:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals