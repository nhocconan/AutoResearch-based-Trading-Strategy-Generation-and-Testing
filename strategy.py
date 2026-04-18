#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) Breakout + Volume Spike + 12h EMA34 Trend Filter
# Donchian breakout captures institutional breakouts with high momentum.
# Volume spike confirms institutional participation.
# 12h EMA34 filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drift.
name = "4h_Donchian20_VolumeSpike_12hEMA34"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA34 on 12h data
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_val = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: Break above upper Donchian band AND price above 12h EMA34 AND volume spike
            if close_val > upper and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: Break below lower Donchian band AND price below 12h EMA34 AND volume spike
            elif close_val < lower and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: Close below 12h EMA34 (trend change) or at lower Donchian band (mean reversion)
            if close_val < ema_val or close_val < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: Close above 12h EMA34 (trend change) or at upper Donchian band (mean reversion)
            if close_val > ema_val or close_val > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals