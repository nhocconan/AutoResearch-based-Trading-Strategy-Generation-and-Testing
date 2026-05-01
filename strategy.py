#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.8x 20-bar MA)
# Camarilla levels provide precise intraday support/resistance, 1d EMA34 filters trend direction, volume spike confirms breakout strength
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) when aligned with higher timeframe trend
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Ranges
    hl_range = high - low
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    # Using previous bar's data to avoid look-ahead
    prev_typical = pd.Series(typical_price).shift(1).values
    prev_hl_range = pd.Series(hl_range).shift(1).values
    
    camarilla_r3 = prev_typical + prev_hl_range * 1.1 / 4.0
    camarilla_s3 = prev_typical - prev_hl_range * 1.1 / 4.0
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA and 1 for shift
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Camarilla breakout conditions
        upper_break = curr_high > camarilla_r3[i]  # Break above R3
        lower_break = curr_low < camarilla_s3[i]   # Break below S3
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3, above 1d EMA34, volume spike
            if upper_break and curr_close > ema_1d_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, below 1d EMA34, volume spike
            elif lower_break and curr_close < ema_1d_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below S3 or below 1d EMA34
            if curr_close < camarilla_s3[i] or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above R3 or above 1d EMA34
            if curr_close > camarilla_r3[i] or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals