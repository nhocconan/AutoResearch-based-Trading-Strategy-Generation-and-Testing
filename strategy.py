#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volatility filter and volume confirmation
# Uses 1d ATR-based volatility regime to avoid whipsaws in low-volatility environments
# Volume spike confirms institutional participation behind the breakout
# Designed for very low frequency (50-150 trades over 4 years) to minimize fee drag
# Works in bull/bear via volatility regime filter + price structure logic

name = "12h_Camarilla_R3S3_Breakout_1dVolFilter_VolumeSpike_v1"
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
    
    # 1d HTF data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR calculation (volatility filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original arrays
    
    # ATR calculation (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period]) / period
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate Camarilla levels from prior 12h bar (using prior bar's HLC)
    prior_high = np.concatenate([[np.nan], high[:-1]])  # prior bar's high
    prior_low = np.concatenate([[np.nan], low[:-1]])    # prior bar's low
    prior_close = np.concatenate([[np.nan], close[:-1]]) # prior bar's close
    
    hl_range = prior_high - prior_low
    camarilla_r3 = prior_close + hl_range * 1.1 / 4
    camarilla_s3 = prior_close - hl_range * 1.1 / 4
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    # Volatility filter: 1d ATR > 20-period average ATR (avoid low-volatility chop)
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > atr_ma_20  # Only trade when volatility is above average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(30, 20)  # Need 1d ATR and volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(atr_aligned[i]) or np.isnan(prior_high[i]) or np.isnan(prior_low[i]) or 
            np.isnan(prior_close[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r3[i]  # Price breaks above R3
        breakout_short = close[i] < camarilla_s3[i]  # Price breaks below S3
        
        # Volatility filter: only trade when 1d ATR is above its 20-period average
        vol_ok = vol_filter[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 with volume spike and sufficient volatility
            if breakout_long and vol_spike and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S3 with volume spike and sufficient volatility
            elif breakout_short and vol_spike and vol_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below prior bar's low or volatility drops significantly
            if close[i] < prior_low[i] or atr[i] < atr_ma_20[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above prior bar's high or volatility drops significantly
            if close[i] > prior_high[i] or atr[i] < atr_ma_20[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals