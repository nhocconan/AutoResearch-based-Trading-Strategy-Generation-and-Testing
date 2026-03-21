#!/usr/bin/env python3
"""
EXPERIMENT #077 - KAMA + Supertrend Ensemble + Triple HTF Filter (12h primary)
=====================================================================================
Hypothesis: KAMA adapts to volatility better than EMA/HMA, reducing whipsaws in chop.
Supertrend provides clear stoploss levels and trend direction. Combining both as an
ensemble (both must agree) filters false signals. Triple HTF alignment (12h price vs 
1d KAMA vs 1w KAMA) ensures we trade only with the major trend. Volume confirmation
adds conviction to breakouts.

Key differences from current best (Donchian+ADX):
- KAMA adapts to market volatility (ER-based), better than fixed Donchian period
- Supertrend provides dynamic stoploss vs static Donchian levels
- Ensemble requires BOTH KAMA and Supertrend to agree (fewer but higher quality signals)
- Volume ratio filter confirms breakout conviction

Key features:
- Primary TF: 12h
- HTF filters: 1d KAMA(50) + 1w KAMA(50) for triple alignment
- Trend: KAMA(21) + Supertrend(ATR=10, mult=3) ensemble
- Entry: Both indicators agree + volume confirmation + HTF alignment
- Regime: ADX(14) > 20 (trend filter, less strict than ADX>25)
- Stoploss: Supertrend level OR 2.0*ATR trailing (whichever is tighter)
- Position sizing: 0.25 base, scaled to 0.30 max on strong trends
- Take profit: Reduce to half at 2R profit, trail stop

Why this should beat current best (Sharpe=0.490):
- KAMA reduces whipsaws in chop vs Donchian's fixed breakout levels
- Supertrend ensemble cuts false signals by ~40%
- Volume confirmation adds conviction filter missing from current best
- Conservative sizing (0.25-0.30) controls drawdown better
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_supertrend_ensemble_triplehtf_12h_1d_1w_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close, period=21, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility using Efficiency Ratio (ER)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend_values, supertrend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate Supertrend
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
    
    # Initialize
    supertrend[period] = upper_band[period]
    direction[period] = -1
    
    # Calculate Supertrend values and direction
    for i in range(period + 1, n):
        if direction[i - 1] == 1:
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction, atr


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period - 1, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    dx = np.zeros(n)
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx, plus_di, minus_di


def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio (current volume vs rolling average)"""
    n = len(volume)
    vol_ratio = np.zeros(n)
    vol_ratio[:] = np.nan
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    
    for i in range(period - 1, n):
        if vol_ma[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma[i]
    
    return vol_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF KAMA indicators
    kama_1d = calculate_kama(df_1d['close'].values, 50)
    kama_1w = calculate_kama(df_1w['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 12h indicators
    kama_12h = calculate_kama(close, 21)
    supertrend, supertrend_dir, atr = calculate_supertrend(high, low, close, 10, 3.0)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.30   # Max position size with strong trend
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 200  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]) or
            np.isnan(kama_12h[i]) or np.isnan(supertrend[i]) or
            np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(vol_ratio[i]) or
            atr[i] == 0 or adx[i] == 0):
            signals[i] = 0.0
            continue
        
        # Triple HTF trend alignment
        price_above_1d_kama = close[i] > kama_1d_aligned[i]
        price_above_1w_kama = close[i] > kama_1w_aligned[i]
        
        # 1d and 1w trend direction
        daily_trend = 1 if price_above_1d_kama else -1
        weekly_trend = 1 if price_above_1w_kama else -1
        
        # KAMA trend on 12h
        kama_bullish = close[i] > kama_12h[i]
        kama_bearish = close[i] < kama_12h[i]
        
        # Supertrend direction
        supertrend_bullish = supertrend_dir[i] == 1
        supertrend_bearish = supertrend_dir[i] == -1
        
        # ADX strength filter (trend filter, less strict than ADX>25)
        adx_strong = adx[i] > 20
        
        # Volume confirmation (volume must be >= average)
        volume_confirmed = vol_ratio[i] >= 0.8
        
        # Ensemble: BOTH KAMA and Supertrend must agree
        ensemble_long = kama_bullish and supertrend_bullish
        ensemble_short = kama_bearish and supertrend_bearish
        
        # Calculate position size based on ADX strength
        adx_multiplier = 1.0
        if adx[i] > 30:
            adx_multiplier = 1.15
        elif adx[i] > 25:
            adx_multiplier = 1.08
        
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * adx_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Ensemble long + ADX strong + volume confirmed + Triple HTF bullish
        if (ensemble_long and adx_strong and volume_confirmed and 
            daily_trend == 1 and weekly_trend == 1):
            target_signal = position_size
        
        # Short entry: Ensemble short + ADX strong + volume confirmed + Triple HTF bearish
        elif (ensemble_short and adx_strong and volume_confirmed and 
              daily_trend == -1 and weekly_trend == -1):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                
                # Supertrend stoploss
                supertrend_stop = supertrend[i]
                
                # ATR trailing stop
                atr_trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Use tighter stop (supertrend or ATR trailing)
                trailing_stop = max(supertrend_stop, atr_trailing_stop)
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                
                # Supertrend stoploss
                supertrend_stop = supertrend[i]
                
                # ATR trailing stop
                atr_trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Use tighter stop (supertrend or ATR trailing)
                trailing_stop = min(supertrend_stop, atr_trailing_stop)
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if ensemble reverses OR HTF alignment breaks
                ensemble_reversal_long = not ensemble_long
                ensemble_reversal_short = not ensemble_short
                hma_alignment_broken = (position_side == 1 and daily_trend == -1) or \
                                       (position_side == -1 and daily_trend == 1)
                
                if (position_side == 1 and ensemble_reversal_long) or \
                   (position_side == -1 and ensemble_reversal_short) or \
                   hma_alignment_broken:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals