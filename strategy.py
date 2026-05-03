#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 with 1d EMA34 uptrend and volume > 2x 20-period MA.
# Short when price breaks below S3 with 1d EMA34 downtrend and volume > 2x 20-period MA.
# Camarilla levels provide precise intraday support/resistance; 1d EMA34 filters for higher-timeframe trend.
# Volume spike confirms institutional participation. Discrete sizing 0.25 to minimize fee churn.
# Target: 75-200 total trades over 4 years (19-50/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA-34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA-34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # We need to shift by 96 bars (4h bars per day) to use previous day's data
    shift_bars = 96
    if n <= shift_bars:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 day)
    high_shift = np.roll(high, shift_bars)
    low_shift = np.roll(low, shift_bars)
    close_shift = np.roll(close, shift_bars)
    
    # Set first shift_bars values to NaN (invalid)
    high_shift[:shift_bars] = np.nan
    low_shift[:shift_bars] = np.nan
    close_shift[:shift_bars] = np.nan
    
    # Calculate Camarilla levels
    range_shift = high_shift - low_shift
    camarilla_multiplier = 1.1 / 12  # R3/S3 multiplier
    
    r3 = close_shift + range_shift * camarilla_multiplier * 3
    s3 = close_shift - range_shift * camarilla_multiplier * 3
    
    # Volume regime: current 4h volume > 2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime from 1d EMA-34
        # Need previous EMA value to determine slope
        if i == 100:
            prev_ema = ema_34_aligned[i-1]
        else:
            prev_ema = ema_34_aligned[i-1]
        
        is_uptrend = ema_val > prev_ema
        is_downtrend = ema_val < prev_ema
        
        # Entry logic
        if position == 0:
            # Long: Price breaks above R3 AND 1d uptrend AND volume spike
            if close_val > r3[i] and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND 1d downtrend AND volume spike
            elif close_val < s3[i] and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S3 OR 1d trend turns down OR volume drops
            if close_val < s3[i] or not is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above R3 OR 1d trend turns up OR volume drops
            if close_val > r3[i] or not is_downtrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals