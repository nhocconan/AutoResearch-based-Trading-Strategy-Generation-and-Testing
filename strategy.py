#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA13) with price > 1d EMA34 and volume spike
# Short when Bear Power < 0 AND Bull Power > 0 with price < 1d EMA34 and volume spike
# Uses 1d EMA34 for higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation ensures momentum participation
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike_v4"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components on 6h: need EMA(13) for Bull/Bear Power
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema_13  # Bull Power = close - EMA13
    bear_power = ema_13 - close  # Bear Power = EMA13 - close (positive when close < EMA13)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_34 = ema_34_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend filter: only long above 1d EMA34, only short below 1d EMA34
        is_uptrend = curr_close > curr_ema_34
        is_downtrend = curr_close < curr_ema_34
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Long: Bull Power > 0 AND Bear Power > 0 (close > EMA13) in uptrend
                # Actually: Bull Power > 0 means close > EMA13, Bear Power < 0 means close > EMA13
                # So both > 0 when close > EMA13
                if curr_bull_power > 0 and curr_bear_power > 0 and is_uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power < 0 AND Bear Power < 0 (close < EMA13) in downtrend
                elif curr_bull_power < 0 and curr_bear_power < 0 and is_downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit when momentum fades
            # Exit when Bull Power turns negative (close < EMA13) OR Bear Power turns positive
            if curr_bull_power <= 0 or curr_bear_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when momentum fades
            # Exit when Bull Power turns positive (close > EMA13) OR Bear Power turns negative
            if curr_bull_power >= 0 or curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals