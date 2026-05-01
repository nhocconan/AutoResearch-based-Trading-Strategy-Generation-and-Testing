#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability breakout zones
# 1d EMA > 50 ensures trend alignment, avoiding false breakouts in chop
# Volume spike confirms institutional participation
# Designed for very low frequency (50-150 trades over 4 years) to minimize fee drag
# Works in bull/bear via trend filter + price structure logic

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation (trend filter)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Camarilla levels from prior 12h bar (using prior bar's HLC)
    prior_high = np.concatenate([[np.nan], high[:-1]])
    prior_low = np.concatenate([[np.nan], low[:-1]])
    prior_close = np.concatenate([[np.nan], close[:-1]])
    
    hl_range = prior_high - prior_low
    camarilla_r3 = prior_close + hl_range * 1.1 / 4
    camarilla_s3 = prior_close - hl_range * 1.1 / 4
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need 1d EMA50 and volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or 
            np.isnan(prior_close[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r3[i]  # Price breaks above R3
        breakout_short = close[i] < camarilla_s3[i]  # Price breaks below S3
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 with volume spike and price above EMA50
            if breakout_long and vol_spike and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S3 with volume spike and price below EMA50
            elif breakout_short and vol_spike and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below prior bar's low or trend reversal (price < EMA50)
            if close[i] < prior_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above prior bar's high or trend reversal (price > EMA50)
            if close[i] > prior_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals