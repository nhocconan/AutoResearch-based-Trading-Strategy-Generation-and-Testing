#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide statistically significant support/resistance from prior day's range.
# Breakouts at R3/S3 with volume and 1d EMA34 trend filter work in both bull and bear markets
# by only taking breakouts aligned with the higher timeframe trend.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate prior 6h bar's high/low for Camarilla levels
    prior_high = pd.Series(high).shift(1).values
    prior_low = pd.Series(low).shift(1).values
    prior_close = pd.Series(close).shift(1).values
    prior_range = prior_high - prior_low
    
    # Camarilla R3 and S3 levels from prior 6h bar
    camarilla_r3 = prior_close + (prior_range * 1.1 / 4)
    camarilla_s3 = prior_close - (prior_range * 1.1 / 4)
    
    # Calculate 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 1, 34, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or np.isnan(prior_close[i]) or
            np.isnan(prior_range[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Camarilla break and 1d EMA34 trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above R3 + price above 1d EMA34
                if curr_close > curr_r3 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below S3 + price below 1d EMA34
                elif curr_close < curr_s3 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: close drops below S3 OR loses 1d trend
            if curr_close < curr_s3 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: close rises above R3 OR loses 1d trend
            if curr_close > curr_r3 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals