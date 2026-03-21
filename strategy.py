#!/usr/bin/env python3
"""
EXPERIMENT #073 - KAMA Adaptive Trend + Volume Confirmation + Dual HTF Filter (15m primary)
============================================================================================
Hypothesis: 15m KAMA adapts to volatility - fast in trends, slow in chop. Combined with
volume confirmation on breakout bars and dual HTF alignment (1h + 4h HMA), this filters
false signals in ranging markets while capturing strong 15m momentum moves.

Key features:
- Primary TF: 15m
- HTF filters: 1h HMA(50) + 4h HMA(50) for dual alignment
- Trend: KAMA(21) adaptive MA + KAMA slope direction
- Volume filter: volume > 1.5x 20-period average on signal bars
- Entry: KAMA crossover + volume spike + HTF alignment
- Regime: ADX(14) > 20 for trend strength
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should beat current best (Sharpe=0.490):
- KAMA adapts to market regime better than static EMA/HMA
- 15m captures more intraday moves than 12h
- Volume filter removes 40%+ of false breakouts
- Dual HTF (1h+4h) simpler than triple, less lag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_volume_dualhtf_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0


def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market volatility - moves fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    for i in range(period, n):
        if i == period:
            kama[i] = close[i]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


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
    """Calculate volume ratio vs rolling average"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio[vol_avg == 0] = 0
    return vol_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_1h = calculate_hma(df_1h['close'].values, 50)
    hma_4h = calculate_hma(df_4h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    kama = calculate_kama(close, period=21)
    atr = calculate_atr(high, low, close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.32   # Max position size with strong signals
    MIN_SIZE = 0.22   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1h_aligned[i]) or np.isnan(hma_4h_aligned[i]) or
            np.isnan(kama[i]) or np.isnan(atr[i]) or np.isnan(adx[i]) or
            np.isnan(vol_ratio[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # Dual HTF trend alignment
        price_above_1h_hma = close[i] > hma_1h_aligned[i]
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        
        # HTF trend direction
        hourly_trend = 1 if price_above_1h_hma else -1
        fourh_trend = 1 if price_above_4h_hma else -1
        
        # KAMA trend direction (slope)
        kama_slope = 0
        if i >= 3 and not np.isnan(kama[i-3]):
            if kama[i] > kama[i-3]:
                kama_slope = 1
            elif kama[i] < kama[i-3]:
                kama_slope = -1
        
        # Price vs KAMA position
        price_above_kama = close[i] > kama[i]
        
        # ADX strength filter (only trade when ADX > 20)
        adx_strong = adx[i] > 20
        
        # Volume confirmation (volume > 1.5x average)
        volume_confirmed = vol_ratio[i] > 1.5
        
        # DI+ vs DI- for trend confirmation
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = minus_di[i] > plus_di[i]
        
        # Calculate position size based on ADX and volume strength
        adx_multiplier = min(1.0 + (adx[i] - 20) / 40, 1.15)  # Max 1.15x
        vol_multiplier = 1.0 + min((vol_ratio[i] - 1.5) / 3, 0.10)  # Max 1.10x
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * adx_multiplier * vol_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Price above KAMA + KAMA slope up + DI+ > DI- + Dual HTF bullish + ADX strong + Volume
        if (price_above_kama and kama_slope == 1 and di_bullish and 
            hourly_trend == 1 and fourh_trend == 1 and adx_strong and volume_confirmed):
            target_signal = position_size
        
        # Short entry: Price below KAMA + KAMA slope down + DI- > DI+ + Dual HTF bearish + ADX strong + Volume
        elif (not price_above_kama and kama_slope == -1 and di_bearish and 
              hourly_trend == -1 and fourh_trend == -1 and adx_strong and volume_confirmed):
            target_signal = -position_size
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
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
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
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
                # Exit if KAMA reverses OR HTF alignment breaks
                kama_reversal_long = (kama_slope == -1 and not price_above_kama)
                kama_reversal_short = (kama_slope == 1 and price_above_kama)
                hma_alignment_broken = (position_side == 1 and hourly_trend == -1) or \
                                       (position_side == -1 and hourly_trend == 1)
                
                if kama_reversal_long or kama_reversal_short or hma_alignment_broken:
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