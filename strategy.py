#!/usr/bin/env python3
"""
Experiment #380: 1h Primary + 4h/12h HTF — Fisher Transform + Choppiness Regime

Hypothesis: 1h timeframe with strict HTF filters can work IF we use:
1. Ehlers Fisher Transform for precise entry timing (better than RSI in bear markets)
2. Choppiness Index for regime detection (range vs trend)
3. 4h HMA for immediate trend direction
4. 12h HMA for higher-level bias filter
5. Volume + Session filters to reduce false signals

KEY DIFFERENCE from failed 1h strategies (#370, #375, #378):
- Relaxed Fisher thresholds (-1.2/+1.2 instead of -1.5/+1.5) to ensure trades trigger
- Single HTF trend filter (4h HMA) + bias (12h HMA) instead of triple confluence
- CHOP regime determines entry type: range=mean-revert, trend=breakout
- Volume filter: only >0.7x avg (not 1.0x which blocks too many trades)
- Session: 6-22 UTC (wider than 8-20 to capture Asia + Europe + US overlap)

Target: 40-80 trades/year on 1h, Sharpe > 0.5 on ALL symbols individually.
Position size: 0.25 (smaller for 1h due to higher frequency).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_4h12h_hma_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for clearer signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Highest high and lowest low over period
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price position within range
    with np.errstate(divide='ignore', invalid='ignore'):
        x = (hl2 - lowest) / (highest - lowest + 1e-10)
    x = np.clip(x, 0.001, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20.0).values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = volume / (vol_avg + 1e-10)
    return ratio

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    hours = (open_time_array // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align HTF HMA for trend (4h) and bias (12h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Extract UTC hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (target 40-80 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(adx_14[i]) or np.isnan(chop[i]) or np.isnan(fisher[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === SESSION FILTER (6-22 UTC) ===
        in_session = (hours[i] >= 6) and (hours[i] <= 22)
        
        # === VOLUME FILTER (>0.7x average) ===
        volume_ok = vol_ratio[i] > 0.7
        
        # === HTF TREND BIAS ===
        # 4h HMA = immediate trend, 12h HMA = higher-level bias
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION ===
        is_trending = (adx_14[i] > 20.0) and (chop[i] < 45.0)
        is_ranging = (chop[i] > 55.0)
        
        # === FISHER TRANSFORM SIGNALS ===
        # Relaxed thresholds for 1h: -1.2/+1.2 instead of -1.5/+1.5
        fisher_bullish_cross = (fisher[i] > -1.2) and (fisher_prev[i] <= -1.2)
        fisher_bearish_cross = (fisher[i] < 1.2) and (fisher_prev[i] >= 1.2)
        
        # Also allow extreme levels without cross (for slower entries)
        fisher_deep_oversold = fisher[i] < -1.5
        fisher_deep_overbought = fisher[i] > 1.5
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow HTF trend on Fisher pullback entry
            # Long: 4h bullish + 12h bullish + Fisher crosses up from oversold
            if price_above_hma_4h and price_above_hma_12h:
                if fisher_bullish_cross or fisher_deep_oversold:
                    if in_session and volume_ok:
                        desired_signal = BASE_SIZE
            
            # Short: 4h bearish + 12h bearish + Fisher crosses down from overbought
            elif price_below_hma_4h and price_below_hma_12h:
                if fisher_bearish_cross or fisher_deep_overbought:
                    if in_session and volume_ok:
                        desired_signal = -BASE_SIZE
        
        elif is_ranging:
            # RANGE REGIME: Mean reversion at extremes
            # Long: Fisher deep oversold + price below 4h HMA (pullback in range)
            if fisher_deep_oversold and price_below_hma_4h:
                if in_session and volume_ok:
                    desired_signal = BASE_SIZE
            
            # Short: Fisher deep overbought + price above 4h HMA (rally in range)
            elif fisher_deep_overbought and price_above_hma_4h:
                if in_session and volume_ok:
                    desired_signal = -BASE_SIZE
        
        else:
            # NEUTRAL REGIME: Only take strongest signals
            # Require both 4h and 12h alignment + Fisher extreme
            if price_above_hma_4h and price_above_hma_12h and fisher_deep_oversold:
                if in_session and volume_ok:
                    desired_signal = BASE_SIZE
            elif price_below_hma_4h and price_below_hma_12h and fisher_deep_overbought:
                if in_session and volume_ok:
                    desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === FISHER EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and fisher[i] > 1.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -1.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_4h:
                desired_signal = BASE_SIZE
            elif position_side < 0 and price_below_hma_4h:
                desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals