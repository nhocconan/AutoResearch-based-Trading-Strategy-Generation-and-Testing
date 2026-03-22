#!/usr/bin/env python3
"""
Experiment #303: 1h Supertrend with 4h HMA Bias and ADX Filter

Hypothesis: After analyzing 299 experiments, the winning formula is clear:
1. 4h Supertrend + 1d HMA (#292) = Sharpe=0.485 (CURRENT BEST)
2. Simple trend + HTF bias works; complex ensembles fail
3. Mean reversion ALWAYS fails across all timeframes
4. 1h timeframe hasn't been properly tested with proven 4h logic

This strategy adapts the winning #292 formula to 1h:
1. 4h HMA(21) for directional bias (proven edge from #292)
2. 1h Supertrend(10, 3.0) for entry timing (classic trend indicator)
3. ADX(14)>15 for trend confirmation (loose threshold for >=10 trades)
4. ATR(14) trailing stoploss at 2.5x (mandatory risk management)
5. Position sizing: 0.25 base, 0.35 in strong trend (discrete levels)

Why this might beat #292:
- 1h captures more trend moves than 4h (more opportunities)
- Supertrend is proven trend-following indicator (works on 4h in #292)
- 4h HMA bias filters out counter-trend trades (same edge as #292)
- Simpler logic = fewer whipsaws than complex ensembles (#297 failed)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_supertrend_4h_hma_adx_atr_v1"
timeframe = "1h"
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
    Supertrend = (HL2) +/- (multiplier * ATR)
    When price > Supertrend = bullish (green)
    When price < Supertrend = bearish (red)
    Returns: supertrend values, direction (1=bullish, -1=bearish)
    """
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    n = len(close)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    # Initialize
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[0] = upper_band[0]
    direction[0] = -1  # Start bearish
    
    for i in range(1, n):
        if direction[i-1] == 1:
            # Previously bullish
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previously bearish
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending, ADX < 20 = ranging.
    We use 15 as threshold for 1h timeframe (looser for more trades).
    """
    n = len(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's method (EMA with span=period)
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25  # Base position size
    SIZE_INCREASED = 0.35  # Increased size in strong trend
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (4h HMA Filter) ===
        # 4h HMA = directional bias (proven edge from #292)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH ===
        # ADX > 15 = trending market (loose threshold for 1h to ensure >=10 trades)
        trending = adx[i] > 15
        strong_trend = adx[i] > 25
        
        # === SUPERTREND SIGNAL ===
        # Supertrend direction: 1 = bullish (price above ST), -1 = bearish
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === VOLATILITY ADJUSTMENT ===
        # Reduce position size when ATR is elevated (>1.5x recent average)
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility and trend strength
        if high_volatility:
            position_size = SIZE_BASE  # Conservative in high vol
        elif strong_trend:
            position_size = SIZE_INCREASED
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: Need 4h bias up + Supertrend bullish + ADX filter
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_4h and  # 4h HMA bias bullish
            st_bullish and  # Supertrend confirms bullish
            trending  # ADX confirms trend
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_4h and  # 4h HMA bias bearish
            st_bearish and  # Supertrend confirms bearish
            trending  # ADX confirms trend
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TREND REVERSAL EXIT ===
        # Exit if HTF bias reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0  # 4h trend reversed against long
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0  # 4h trend reversed against short
        
        # === SUPERTREND REVERSAL EXIT ===
        # Exit if Supertrend reverses against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and st_bearish:
                new_signal = 0.0  # Supertrend broke against long
            if position_side < 0 and st_bullish:
                new_signal = 0.0  # Supertrend broke against short
        
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
            # else: maintaining same position direction (possibly adjusted size)
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