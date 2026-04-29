#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with volume spike and 1d EMA34 trend filter
# Uses wider Camarilla levels (R4/S4 = C ± (H-L)*1.5/4) for fewer, higher-quality breakouts
# Volume spike (>2.0x 30-period average) confirms strong participation
# 1d EMA34 trend filter ensures alignment with daily trend to avoid counter-trend whipsaws
# Designed for 12h timeframe: targets 12-37 trades/year (50-150 total over 4 years)
# Works in bull/bear: volume confirms breakout validity, 1d EMA34 filters noise

name = "12h_Camarilla_R4S4_VolumeSpike_1dEMA34_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R4/S4 = C ± (H-L)*1.5/4.0 (wider bands for fewer trades)
    camarilla_range = (prev_high - prev_low) * 1.5 / 4.0
    r4 = prev_close + camarilla_range
    s4 = prev_close - camarilla_range
    
    # Align daily levels to 12h timeframe (wait for daily bar to close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 2.0x 30-period average (stricter for fewer trades)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 34)  # warmup for volume MA and 1d EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_30[i]) or np.isnan(ema_34_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r4 = r4_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34 = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above R4 with volume and above 1d EMA34
                if curr_high > curr_r4 and curr_close > curr_ema_34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S4 with volume and below 1d EMA34
                elif curr_low < curr_s4 and curr_close < curr_ema_34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below S4 (reversal signal)
            if curr_low < curr_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above R4 (reversal signal)
            if curr_high > curr_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals