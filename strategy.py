#!/usr/bin/env python3
"""
6h_MarketPhase_Rotation
Hypothesis: Combines 12h market phase detection (via RSI divergence and volume) with 6s entry timing using price action at key levels. In bull phases, buy pullbacks to VWAP; in bear phases, sell rallies to VWAP; in neutral, fade extremes. Uses volume-weighted price action to avoid whipsaws. Designed for low trade frequency (15-25/year) by requiring confluence of phase, volume, and price action.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for market phase detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h RSI for phase detection
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h[:14] = np.nan
    
    # Calculate 12h volume trend (20-period EMA)
    vol_12h = df_12h['volume'].values
    vol_ema_20 = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_increasing = vol_12h > vol_ema_20
    
    # Market phase: bull if RSI > 50 and volume increasing, bear if RSI < 50 and volume increasing, neutral otherwise
    bull_phase = (rsi_12h > 50) & vol_increasing
    bear_phase = (rsi_12h < 50) & vol_increasing
    neutral_phase = ~(bull_phase | bear_phase)
    
    # Align phase indicators to 6s
    bull_phase_aligned = align_htf_to_ltf(prices, df_12h, bull_phase.astype(float))
    bear_phase_aligned = align_htf_to_ltf(prices, df_12h, bear_phase.astype(float))
    neutral_phase_aligned = align_htf_to_ltf(prices, df_12h, neutral_phase.astype(float))
    
    # Calculate 6s VWAP (typical price * volume)
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # VWAP deviation bands (1.5 * ATR)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    vwap_upper = vwap + 1.5 * atr
    vwap_lower = vwap - 1.5 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_phase_aligned[i]) or 
            np.isnan(bear_phase_aligned[i]) or
            np.isnan(neutral_phase_aligned[i]) or
            np.isnan(vwap[i]) or
            np.isnan(vwap_upper[i]) or
            np.isnan(vwap_lower[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Price relative to VWAP bands
        price_above_vwap_upper = close[i] > vwap_upper[i]
        price_below_vwap_lower = close[i] < vwap_lower[i]
        price_near_vwap = (close[i] >= vwap_lower[i]) & (close[i] <= vwap_upper[i])
        
        # Entry logic based on market phase
        long_entry = False
        short_entry = False
        
        if bull_phase_aligned[i] > 0.5:  # Bull phase
            # Buy pullbacks to VWAP in uptrend
            long_entry = price_near_vwap and (close[i] > close[i-1]) and (position <= 0)
        elif bear_phase_aligned[i] > 0.5:  # Bear phase
            # Sell rallies to VWAP in downtrend
            short_entry = price_near_vwap and (close[i] < close[i-1]) and (position >= 0)
        else:  # Neutral phase
            # Fade extremes in ranging market
            long_entry = price_below_vwap_lower and (position <= 0)
            short_entry = price_above_vwap_upper and (position >= 0)
        
        # Exit conditions: opposite signal or extreme reversal
        long_exit = position == 1 and (price_above_vwap_upper or (bear_phase_aligned[i] > 0.5 and close[i] < vwap[i]))
        short_exit = position == -1 and (price_below_vwap_lower or (bull_phase_aligned[i] > 0.5 and close[i] > vwap[i]))
        
        if long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        elif long_entry:
            signals[i] = 0.25
            position = 1
        elif short_entry:
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_MarketPhase_Rotation"
timeframe = "6h"
leverage = 1.0