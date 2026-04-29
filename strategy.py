#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume spike
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA of close)
# Long when Bull Power > 0 and rising, price above 1d EMA34, volume > 2x average
# Short when Bear Power < 0 and falling, price below 1d EMA34, volume > 2x average
# Works in bull/bear: 1d EMA34 filters counter-trend trades, Elder Ray shows momentum exhaustion
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_ElderRay_VolumeSpike_1dEMA34_Trend_v1"
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
    
    # Calculate 13-period EMA for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34, 13)  # warmup for volume MA, 1d EMA, and EMA13
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_30[i]) or np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34 = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: Bull Power > 0 and rising (momentum building)
                if curr_bull_power > 0 and curr_bull_power > bull_power[i-1] and curr_close > curr_ema_34:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power < 0 and falling (momentum building)
                elif curr_bear_power < 0 and curr_bear_power < bear_power[i-1] and curr_close < curr_ema_34:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Bull Power turns negative (momentum lost)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Bear Power turns positive (momentum lost)
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals