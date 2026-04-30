#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA34 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13; Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1d EMA34 ensures alignment with daily trend; volume spike >2.0x confirms participation
# Works in bull/bear: Bull Power > 0 + rising indicates bullish momentum, Bear Power < 0 + falling indicates bearish momentum
# Discrete sizing (0.25) targets 50-150 total trades over 4 years to avoid fee drag

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 30, 13, 34)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(ema13[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: Bull Power > 0 and rising + above 1d EMA34
                if curr_bull_power > 0 and curr_bull_power > bull_power[i-1] and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power < 0 and falling + below 1d EMA34
                elif curr_bear_power < 0 and curr_bear_power < bear_power[i-1] and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative or Bear Power becomes positive
            if curr_bull_power <= 0 or curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive or Bull Power becomes negative
            if curr_bear_power >= 0 or curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals