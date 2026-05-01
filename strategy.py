#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1w EMA(50) > EMA(200) as trend filter (bullish regime) + Camarilla breakout from prior 1d bar
# Volume spike (2.0x 20-period MA) confirms participation
# Designed for very low frequency (30-100 trades over 4 years) to minimize fee drag on 1d timeframe
# Works in bull markets via trend filter - avoids chop and bear markets by only taking longs in strong uptrends

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # 1w EMA(50) and EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Calculate Camarilla levels from prior 1d bar (using prior day's HLC)
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
    position = 0  # 0: flat, 1: long
    
    # Start after warmup for all indicators
    start_idx = max(200, 20)  # Need EMA200 and volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or 
            np.isnan(prior_close[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA50 > EMA200 indicates bullish regime
        bullish_regime = ema_50_aligned[i] > ema_200_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r3[i]  # Price breaks above R3
        # Note: No short entries - only long in bullish regime
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new long entries
            # Long: Breakout above R3 with volume spike and bullish regime
            if breakout_long and vol_spike and bullish_regime:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below prior day's low or trend reversal (EMA50 < EMA200)
            if close[i] < prior_low[i] or ema_50_aligned[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals