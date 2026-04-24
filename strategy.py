#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA34 trend filter + volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter.
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close).
- Entry: Long when Bull Power > 0 AND Bear Power rising (trough) AND 1d EMA34 uptrend AND volume > 2.0 * 6h volume MA(50);
         Short when Bear Power < 0 AND Bull Power falling (peak) AND 1d EMA34 downtrend AND volume > 2.0 * 6h volume MA(50).
- Exit: Close-based reversal or trend change (signal=0 when 1d EMA34 slope changes sign).
- Signal size: 0.25 discrete to minimize fee drag.
- Elder Ray measures bull/bear strength relative to EMA13; rising Bull Power in uptrend shows accumulation; falling Bear Power in downtrend shows distribution.
- Works in bull markets (buy strength on pullbacks) and bear markets (sell weakness on rallies) with trend filter to avoid counter-trend trades.
- Volume spike (2.0x) confirms institutional participation. Estimated trades: ~80 total over 4 years (~20/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34_1d - np.roll(ema_34_1d, 1)
    ema_34_slope[0] = 0
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Slope of Elder Ray components to detect turning points
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    bull_power_slope[0] = 0
    bear_power_slope[0] = 0
    
    # Volume confirmation: 2.0x 50-period MA
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align all HTF and LTF indicators to primary 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    ema_13_aligned = align_htf_to_ltf(prices, prices, ema_13)  # same timeframe
    bull_power_aligned = align_htf_to_ltf(prices, prices, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, prices, bear_power)
    bull_power_slope_aligned = align_htf_to_ltf(prices, prices, bull_power_slope)
    bear_power_slope_aligned = align_htf_to_ltf(prices, prices, bear_power_slope)
    vol_ma_50_aligned = align_htf_to_ltf(prices, prices, vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for EMA34, EMA13, and volume MA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_slope_aligned[i]) or 
            np.isnan(ema_13_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(bull_power_slope_aligned[i]) or np.isnan(bear_power_slope_aligned[i]) or np.isnan(vol_ma_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Exit: trend change (1d EMA34 slope changes sign)
        if position != 0:
            if position == 1 and ema_34_slope_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and ema_34_slope_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions
        # Bullish: Bull Power > 0 (strong bulls) AND Bear Power rising from trough (bulls gaining)
        # Bearish: Bear Power < 0 (strong bears) AND Bull Power falling from peak (bears gaining)
        bullish_setup = (bull_power_aligned[i] > 0) and (bear_power_slope_aligned[i] > 0)
        bearish_setup = (bear_power_aligned[i] < 0) and (bull_power_slope_aligned[i] < 0)
        
        # Trend filter: only trade in direction of 1d EMA34 slope
        uptrend = ema_34_slope_aligned[i] > 0
        downtrend = ema_34_slope_aligned[i] < 0
        
        # Volume confirmation (2.0x average volume)
        vol_confirm = volume[i] > 2.0 * vol_ma_50_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Bull Power > 0 AND Bear Power rising AND uptrend
                if bullish_setup and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 AND Bull Power falling AND downtrend
                elif bearish_setup and downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0