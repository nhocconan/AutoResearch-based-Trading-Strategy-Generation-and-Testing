#!/usr/bin/env python3
"""
Experiment #335: 12h Volatility Breakout with 1d HMA Bias and Supertrend

Hypothesis: After analyzing 286 failed strategies and the experiment history:
1. #329 (12h MACD + dual HTF) achieved Sharpe=0.311 - shows 12h CAN work
2. Most failures come from too many conflicting filters (0 trades)
3. Mean reversion fails on crypto, but volatility breakouts work
4. 12h timeframe = fewer trades but higher quality, less noise

Key improvements:
1. Volatility expansion filter (ATR7/ATR30 > 1.2) - catches breakout momentum
2. Donchian(20) breakout - proven breakout signal
3. Supertrend(10,3) - dynamic trend confirmation
4. 1d HMA bias - higher timeframe directional filter
5. ADX > 20 + CHOP < 50 - ensures trending regime, not range
6. Position size 0.30 with 2x ATR trailing stop

Why this should work on 12h:
- 12h captures multi-day moves without intraday noise
- Volatility expansion precedes major moves (learned from vol spike research)
- Donchian breakout is simple but effective when filtered properly
- Supertrend provides dynamic support/resistance
- 1d HMA gives stable trend bias (proven in successful strategies)
- Loose enough filters to generate ≥10 trades but strict enough for quality

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vol_breakout_1d_hma_supertrend_donchian_atr_v1"
timeframe = "12h"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    supertrend = np.full(n, np.nan)
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Calculate Supertrend
    supertrend[period] = upper_band[period]
    for i in range(period + 1, n):
        if np.isnan(atr[i]):
            supertrend[i] = np.nan
            continue
        if close[i] > supertrend[i - 1]:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Determine direction
    direction = np.where(close > supertrend, 1, -1)
    direction[:period] = np.nan
    
    return supertrend, direction

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        tr_sum = 0.0
        for j in range(1, period + 1):
            tr_sum += np.abs(high[i - j + 1] - low[i - j + 1])
        
        if tr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    adx = calculate_adx(high, low, close, 14)
    chop = calculate_chop(high, low, close, 14)
    
    # Volatility ratio
    vol_ratio = np.full(n, np.nan)
    vol_ratio[30:] = atr_7[30:] / np.maximum(atr_30[30:], 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
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
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1d HMA = primary directional bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === SUPERTREND DIRECTION ===
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i - 1]
        donchian_breakout_down = close[i] < donchian_lower[i - 1]
        
        # === ADX TREND STRENGTH ===
        trending = adx[i] > 20
        
        # === CHOPPINESS FILTER ===
        not_choppy = chop[i] < 50
        
        # === VOLATILITY EXPANSION ===
        vol_expanding = vol_ratio[i] > 1.2
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG: 1d trend up + Supertrend bullish + Donchian breakout + trending + not choppy + vol expanding
        long_conditions = (
            bull_trend_1d and
            st_bullish and
            donchian_breakout_up and
            trending and
            not_choppy and
            vol_expanding
        )
        
        # SHORT: 1d trend down + Supertrend bearish + Donchian breakout + trending + not choppy + vol expanding
        short_conditions = (
            bear_trend_1d and
            st_bearish and
            donchian_breakout_down and
            trending and
            not_choppy and
            vol_expanding
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = SIZE
        
        if short_conditions:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 1d trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1d:
                new_signal = 0.0
            if position_side < 0 and bull_trend_1d:
                new_signal = 0.0
        
        # === SUPERTREND REVERSAL EXIT ===
        # Exit if Supertrend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and st_bearish:
                new_signal = 0.0
            if position_side < 0 and st_bullish:
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