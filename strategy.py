#!/usr/bin/env python3
"""
Experiment #304: 4h Supertrend with Dual HTF HMA Bias and Regime Filter

Hypothesis: Building on #292 (Sharpe=0.485), this strategy adds:
1. 1d HMA(21) for primary directional bias (proven edge)
2. 1w HMA(21) for meta-trend filter (reduces false signals in chop)
3. 4h Supertrend(10,3) for entry timing (catches trends early)
4. Bollinger Band Width regime filter (avoid trading in choppy markets)
5. ADX(14)>12 for trend confirmation (loose threshold for >=10 trades)
6. ATR(14) trailing stoploss at 2.5x (tighter than 3.5x for better DD control)

Why this might beat #292:
- Regime filter (BB Width) avoids whipsaws in ranging markets
- Dual HTF (1d + 1w) provides stronger trend confirmation
- Looser ADX threshold ensures sufficient trade generation
- Tighter stoploss reduces drawdown during reversals

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_supertrend_dual_htf_regime_adx_atr_v1"
timeframe = "4h"
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
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    upper_band[:] = np.nan
    lower_band[:] = np.nan
    supertrend[:] = np.nan
    direction[:] = np.nan
    
    for i in range(period, n):
        if np.isnan(atr[i]):
            continue
        
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            if close[i - 1] <= supertrend[i - 1]:
                supertrend[i] = min(upper_band[i], supertrend[i - 1])
                if close[i] <= supertrend[i]:
                    direction[i] = -1
                else:
                    direction[i] = 1
                    supertrend[i] = lower_band[i]
            else:
                supertrend[i] = max(lower_band[i], supertrend[i - 1])
                if close[i] >= supertrend[i]:
                    direction[i] = 1
                else:
                    direction[i] = -1
                    supertrend[i] = upper_band[i]
    
    return supertrend, direction

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

def calculate_bb_width(close, period=20, std_dev=2.0):
    """
    Calculate Bollinger Band Width for regime detection.
    BB Width = (Upper - Lower) / Middle
    Low BB Width = squeeze/consolidation, High BB Width = expansion/trend
    """
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    
    bb_width = (upper - lower) / middle
    return bb_width.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    adx = calculate_adx(high, low, close, 14)
    bb_width = calculate_bb_width(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    # Calculate BB Width percentile for regime filter
    bb_width_percentile = np.zeros(n)
    for i in range(100, n):
        if np.isnan(bb_width[i]):
            bb_width_percentile[i] = np.nan
            continue
        recent_bw = bb_width[max(0, i-100):i+1]
        recent_bw = recent_bw[~np.isnan(recent_bw)]
        if len(recent_bw) > 0:
            bb_width_percentile[i] = np.sum(bb_width[i] >= recent_bw) / len(recent_bw)
        else:
            bb_width_percentile[i] = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_percentile[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS (Dual HTF Filter) ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH ===
        # ADX > 12 = trending market (loose threshold for 4h to ensure trades)
        trending = adx[i] > 12
        strong_trend = adx[i] > 20
        
        # === REGIME FILTER (Bollinger Band Width) ===
        # BB Width percentile > 0.3 = not in extreme squeeze (avoid chop)
        # This filters out low-volatility consolidation periods
        not_squeeze = bb_width_percentile[i] > 0.25
        
        # === SUPERTREND SIGNAL ===
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === VOLATILITY ADJUSTMENT ===
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size based on volatility and trend strength
        if high_volatility:
            position_size = SIZE_BASE
        elif strong_trend:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG ENTRY: 1d bias up + 1w bias up + Supertrend bullish + ADX + Regime OK
        # Looser conditions to ensure >=10 trades per symbol
        long_conditions = (
            bull_trend_1d and
            bull_trend_1w and
            st_bullish and
            trending and
            not_squeeze
        )
        
        # SHORT ENTRY: Mirror of long
        short_conditions = (
            bear_trend_1d and
            bear_trend_1w and
            st_bearish and
            trending and
            not_squeeze
        )
        
        # === GENERATE SIGNAL ===
        if long_conditions:
            new_signal = position_size
        
        if short_conditions:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and (bear_trend_1d or bear_trend_1w):
                new_signal = 0.0
            if position_side < 0 and (bull_trend_1d or bull_trend_1w):
                new_signal = 0.0
        
        # === SUPERTREND REVERSAL EXIT ===
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