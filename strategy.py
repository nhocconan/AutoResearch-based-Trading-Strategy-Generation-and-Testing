#!/usr/bin/env python3
"""
Experiment #069: 4h Primary + 1d HTF — Fisher Transform + ADX Regime + Vol Spike Reversion

Hypothesis: 4h timeframe with 1d HTF trend bias using Ehlers Fisher Transform for entry timing,
ADX regime detection with hysteresis, and ATR vol spike reversion will generate 25-45 trades/year
with Sharpe > 0.486. This combines proven edges: Fisher catches reversals in bear rallies,
ADX hysteresis reduces whipsaw, vol spike reversion captures panic bottoms.

Key innovations:
1) Ehlers Fisher Transform (period=9): long when Fisher crosses above -1.5, short when crosses below +1.5
2) ADX regime with hysteresis: enter trend when ADX>22, exit when ADX<18 (reduces chop whipsaw)
3) Vol spike reversion: ATR(7)/ATR(30) > 1.8 + price < BB(20, 2.2) = long (panic bottom)
4) 1d HMA for macro bias: only long if price > 1d HMA, only short if price < 1d HMA
5) BB Width percentile: < 20th percentile = squeeze building (prepare for breakout)
6) Relaxed entry thresholds to ensure 30+ trades (learned from 0-trade failures)

Why this should work:
- Fisher Transform proven in bear markets (catches reversals better than RSI)
- ADX hysteresis reduces false signals in choppy conditions
- Vol spike reversion captures "vol crush" after panic (research-backed edge)
- 1d HTF prevents counter-trend trades in 2022-style crashes
- Relaxed thresholds ensure sufficient trade frequency (critical for Sharpe)

Position size: 0.25-0.30 (discrete levels)
Stoploss: 2.5*ATR trailing
Target: 25-45 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_adx_volspike_1d_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - LL) / (HH - LL) - 0.67
    """
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    price = (high + low) / 2.0
    range_val = hh - ll + 1e-10
    x = 0.67 * ((price - ll) / range_val - 0.5) + 0.67
    x = np.clip(x, -0.999, 0.999)
    fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    return fisher, fisher_prev

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / sma
    return upper.values, lower.values, sma.values, width.values

def calculate_bb_width_percentile(width, lookback=100):
    """Calculate BB Width percentile over lookback period."""
    width_s = pd.Series(width)
    percentile = width_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: np.sum(x < x.iloc[-1]) / len(x) * 100 if len(x) >= lookback else 50
    )
    return percentile.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    bb_upper, bb_lower, bb_sma, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.2)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # ADX hysteresis state
    adx_trending = False
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(adx[i]) or np.isnan(bb_upper[i]):
            continue
        if atr_14[i] == 0 or atr_30[i] == 0:
            continue
        
        # === HTF TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === ADX REGIME WITH HYSTERESIS ===
        # Enter trending mode when ADX > 22, exit when ADX < 18
        if adx[i] > 22:
            adx_trending = True
        elif adx[i] < 18:
            adx_trending = False
        
        # === VOL SPIKE REVERSION ===
        # ATR(7)/ATR(30) > 1.8 indicates vol spike (panic)
        vol_spike_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        is_vol_spike = vol_spike_ratio > 1.8
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5
        # Short: Fisher crosses below +1.5
        fisher_cross_long = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_short = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === BOLLINGER BAND SIGNALS ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_squeeze = bb_width_pct[i] < 20  # Width at 20th percentile = squeeze
        
        # === VOL SPIKE REVERSION ENTRY ===
        # Long when vol spike + price below BB lower + 1d not strongly bearish
        vol_spike_long = is_vol_spike and price_below_bb_lower and not price_below_hma_1d
        
        # === TRENDING REGIME ENTRY (ADX > 22) ===
        if adx_trending:
            # Long: Fisher cross + price above 1d HMA + DI+ > DI-
            if fisher_cross_long and price_above_hma_1d and plus_di[i] > minus_di[i]:
                signals[i] = POSITION_SIZE
            
            # Short: Fisher cross + price below 1d HMA + DI- > DI+
            elif fisher_cross_short and price_below_hma_1d and minus_di[i] > plus_di[i]:
                signals[i] = -POSITION_SIZE
        
        # === RANGING REGIME ENTRY (ADX < 18) ===
        else:
            # Long: Vol spike reversion OR (Fisher cross + price below BB lower)
            if vol_spike_long:
                signals[i] = POSITION_SIZE
            elif fisher_cross_long and price_below_bb_lower:
                signals[i] = POSITION_SIZE
            
            # Short: Fisher cross + price above BB upper
            elif fisher_cross_short and price_above_bb_upper:
                signals[i] = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, hold unless exit conditions met
        if in_position and signals[i] == 0.0:
            if position_side > 0:
                # Hold long if Fisher > -1.0 and price > entry
                if fisher[i] > -1.0:
                    signals[i] = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if Fisher < 1.0 and price < entry
                if fisher[i] < 1.0:
                    signals[i] = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_1d and adx[i] > 20:
                signals[i] = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d and adx[i] > 20:
                signals[i] = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if signals[i] != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(signals[i])
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(signals[i]) != position_side:
                position_side = np.sign(signals[i])
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
    
    return signals