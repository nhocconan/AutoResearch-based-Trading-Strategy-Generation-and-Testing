#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 Trend Filter and Volume Confirmation
# Elder Ray measures bull/bear power relative to EMA13 - identifies strength of buyers/sellers
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and increasing + price above 1d EMA34 (uptrend)
# Short when Bear Power < 0 and decreasing + price below 1d EMA34 (downtrend)
# Volume confirmation (>1.5x average) ensures conviction
# Works in bull/bear: trend filter aligns with higher timeframe, Elder Ray captures momentum shifts
# Discrete sizing (0.25) targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (13-period EMA of close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull power: high minus EMA13
    bear_power = low - ema13   # Bear power: low minus EMA13
    
    # Rate of change of Elder Ray to detect strengthening/weakening
    bull_power_change = np.diff(bull_power, prepend=bull_power[0])
    bear_power_change = np.diff(bear_power, prepend=bear_power[0])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or
            np.isnan(bull_power_change[i]) or
            np.isnan(bear_power_change[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_bull_power_change = bull_power_change[i]
        curr_bear_power_change = bear_power_change[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Long: Bull power positive AND increasing + price above 1d EMA34 (uptrend)
                if curr_bull_power > 0 and curr_bull_power_change > 0 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear power negative AND decreasing + price below 1d EMA34 (downtrend)
                elif curr_bear_power < 0 and curr_bear_power_change < 0 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Bull power turns negative OR price closes below 1d EMA34 (trend change)
            if curr_bull_power <= 0 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear power turns positive OR price closes above 1d EMA34 (trend change)
            if curr_bear_power >= 0 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals