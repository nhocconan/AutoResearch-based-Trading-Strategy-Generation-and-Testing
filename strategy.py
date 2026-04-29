#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with Daily EMA34 Trend Filter and Volume Spike
# Camarilla pivot levels (R3/S3) act as strong support/resistance - breaks indicate institutional participation
# Daily EMA34 ensures we trade breakouts in direction of higher timeframe trend (works in bull/bear)
# Volume spike (>2x average) confirms breakout validity and reduces false signals
# Target: 25-40 trades/year (100-160 total over 4 years) - tight entries to minimize fee drag

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from previous day (using daily data)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use previous day's OHLC to calculate today's levels (no look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_raw = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (using previous day's data)
    camarilla_r3 = np.full(len(close_1d_raw), np.nan)
    camarilla_s3 = np.full(len(close_1d_raw), np.nan)
    
    for i in range(1, len(close_1d_raw)):
        # Use previous day's OHLC to calculate today's Camarilla levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d_raw[i-1]
        rang = prev_high - prev_low
        
        camarilla_r3[i] = prev_close + (rang * 1.1 / 4)
        camarilla_s3[i] = prev_close - (rang * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # warmup for daily EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_ema34_1d = ema34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Determine trend regime from daily EMA34
        bullish_regime = curr_close > curr_ema34_1d
        bearish_regime = curr_close < curr_ema34_1d
        
        if position == 0:  # Flat - look for new entries
            # Look for Camarilla level breakouts with volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above R3 in bullish regime
                if bullish_regime and curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S3 in bearish regime
                elif bearish_regime and curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price crosses below S3 (reversal) OR volume dies off
            if curr_close < curr_s3 or not curr_volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price crosses above R3 (reversal) OR volume dies off
            if curr_close > curr_r3 or not curr_volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals