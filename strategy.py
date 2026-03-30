#!/usr/bin/env python3
"""
Experiment #024: 6h Donchian Breakout + Weekly VWAP Trend + Volume (6h)

HYPOTHESIS: 6h timeframe balances between too few trades (12h/1d) and overtrading (4h).
Weekly VWAP provides structural trend direction without being too restrictive.
Donchian(20) captures medium-term breakouts every 20-40 bars. Volume spike confirms.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Breakout above 20-bar high + volume spike + above weekly VWAP = strong momentum
- Bear: Breakdown below 20-bar low + volume spike + below weekly VWAP = strong short
- Weekly VWAP acts as regime filter: prevents fighting major trend

EXPECTED TRADES: 75-150 total over 4 years (19-37/year per symbol)
- Donchian(20) on 6h = break every ~20-40 bars = 219-438 potential/year
- Volume spike (1.5x) → reduces by ~40%
- Weekly VWAP trend filter → reduces by ~30%
- Final: ~75-150 trades = statistical validity
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_vwap_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_vwap(high, low, close, volume):
    """Weekly VWAP - anchored to week start"""
    typical_price = (high + low + close) / 3.0
    cumvol = pd.Series(volume).cumsum()
    cumtpv = pd.Series(typical_price * volume).cumsum()
    vwap = cumtpv / cumvol
    return vwap.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly close for trend
    weekly_close = df_1w['close'].values
    weekly_vwap = calculate_vwap(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values,
        df_1w['volume'].values
    )
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, weekly_vwap)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Local 6h VWAP for trend
    local_vwap = calculate_vwap(high, low, close, volume)
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Enough for Donchian20, ATR14, VWAP
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(weekly_vwap_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: Weekly VWAP + local confirmation ===
        prev_wvwap = weekly_vwap_aligned[i-1] if i > 0 and not np.isnan(weekly_vwap_aligned[i-1]) else weekly_vwap_aligned[i]
        
        above_weekly = close[i] > weekly_vwap_aligned[i]
        above_local = close[i] > local_vwap[i]
        bull_trend = above_weekly and above_local
        
        below_weekly = close[i] < weekly_vwap_aligned[i]
        below_local = close[i] < local_vwap[i]
        bear_trend = below_weekly and below_local
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           high[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           low[i] < prev_donchian_low)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend
            if bullish_breakout and vol_spike and bull_trend:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend
            elif bearish_breakout and vol_spike and bear_trend:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.5 ATR from highest
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend flips
                elif close[i] < weekly_vwap_aligned[i] * 0.995:  # slight buffer
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.5 ATR from lowest
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend flips
                elif close[i] > weekly_vwap_aligned[i] * 1.005:  # slight buffer
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals