#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1w trend filter and 1d volume confirmation
# - Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Long when Bull Power > 0 AND 1w EMA(34) rising (trend up) AND 1d volume > 1.5x 20-period average
# - Short when Bear Power < 0 AND 1w EMA(34) falling (trend down) AND 1d volume > 1.5x 20-period average
# - Exit when power crosses zero (reversal signal)
# - Uses discrete position sizing 0.25 to minimize fee churn
# - Target: 12-37 trades/year on 6h (50-150 total over 4 years)
# - Works in bull/bear: 1w EMA defines primary trend, Elder Ray measures momentum within trend,
#   volume confirms institutional participation

name = "6h_1w_1d_elderray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 40 or len(df_1d) < 40:
        return np.zeros(n)
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_6h = prices['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 6h Bull Power and Bear Power
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13
    bear_power = low_6h - ema_13
    
    # Pre-compute 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, prices, bull_power)  # 6h data already in prices
    bear_power_aligned = align_htf_to_ltf(prices, prices, bear_power)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    prev_bull_power = np.full(n, np.nan)  # for crossover detection
    prev_bear_power = np.full(n, np.nan)
    
    for i in range(100, n):  # Start after warmup
        # Store previous powers for crossover detection
        if i > 0:
            prev_bull_power[i] = bull_power_aligned[i-1]
            prev_bear_power[i] = bear_power_aligned[i-1]
        else:
            prev_bull_power[i] = np.nan
            prev_bear_power[i] = np.nan
        
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(prev_bull_power[i]) or np.isnan(prev_bear_power[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition (1.5x average)
        vol_series = prices['volume'].values
        vol_ma_6h = pd.Series(vol_series).rolling(window=20, min_periods=20).mean().values
        vol_spike = not np.isnan(vol_ma_6h[i]) and vol_series[i] > 1.5 * vol_ma_6h[i]
        
        bull_now = bull_power_aligned[i]
        bull_prev = prev_bull_power[i]
        bear_now = bear_power_aligned[i]
        bear_prev = prev_bear_power[i]
        ema_34_now = ema_34_1w_aligned[i]
        ema_34_prev = ema_34_1w_aligned[i-1] if i > 0 else np.nan
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND 1w EMA rising AND volume spike
            if (bull_now > 0 and not np.isnan(ema_34_prev) and ema_34_now > ema_34_prev and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power < 0 AND 1w EMA falling AND volume spike
            elif (bear_now < 0 and not np.isnan(ema_34_prev) and ema_34_now < ema_34_prev and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Power crosses zero (reversal signal)
            exit_long = (position == 1 and bull_prev > 0 and bull_now <= 0)
            exit_short = (position == -1 and bear_prev < 0 and bear_now >= 0)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals