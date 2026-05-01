#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from prior 1d to identify key support/resistance (R3/S3 for breakout, R4/S4 for continuation)
# EMA34 on 1d determines primary trend direction (only take longs above EMA34, shorts below)
# Volume confirmation > 2.0x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~15-30 trades/year per symbol with 0.25 sizing
# Camarilla R3/S3 breakouts have high follow-through in trending markets, especially with volume confirmation
# EMA34 filter avoids counter-trend trades during strong trends, improving win rate in both bull and bear markets

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v1"
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
    
    # 1d HTF data for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Camarilla levels: based on prior day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's values (shifted by 1 to avoid look-ahead)
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Camarilla calculations
    range_ = prior_high - prior_low
    camarilla_h5 = prior_close + (range_ * 1.1 / 2)  # R4 equivalent
    camarilla_h4 = prior_close + (range_ * 1.1 / 4)  # R3
    camarilla_h3 = prior_close + (range_ * 1.1 / 6)  # R2
    camarilla_l3 = prior_close - (range_ * 1.1 / 6)  # S2
    camarilla_l4 = prior_close - (range_ * 1.1 / 4)  # S3
    camarilla_l5 = prior_close - (range_ * 1.1 / 2)  # S4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d EMA34 (34 days) + volume EMA20
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA34
        bullish_trend = close[i] > ema_34_aligned[i]
        bearish_trend = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_trend:
                # Long: break above Camarilla H3 (R2) with volume spike, target H4 (R3)
                if close[i] > camarilla_h3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_trend:
                # Short: break below Camarilla L3 (S2) with volume spike, target L4 (S3)
                if close[i] < camarilla_l3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA
        
        elif position == 1:  # Long position
            # Exit: close below Camarilla L3 (S2) or reverse signal with volume
            if close[i] < camarilla_l3_aligned[i] and volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: close above Camarilla H3 (R2) or reverse signal with volume
            if close[i] > camarilla_h3_aligned[i] and volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals