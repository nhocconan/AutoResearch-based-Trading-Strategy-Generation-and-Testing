#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla R3/S3 levels represent strong intraday support/resistance - breaks indicate institutional participation
# 1d EMA34 provides medium-term trend filter to align with higher timeframe momentum
# Volume spike (>2.0x average) confirms breakout legitimacy and filters false signals
# Works in bull/bear: breakouts occur in all regimes, volume confirms legitimacy, trend filter reduces false signals
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Calculate Camarilla pivot levels for 6h using previous bar's OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_high = prev_close + ((prev_high - prev_low) * 1.1 / 4)   # R3
    camarilla_low = prev_close - ((prev_high - prev_low) * 1.1 / 4)    # S3
    
    # Breakout conditions
    breakout_up = close > camarilla_high
    breakout_down = close < camarilla_low
    
    # Volume spike confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_high[i]) or 
            np.isnan(camarilla_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume spike and trend filter
            if curr_volume_spike:
                # Bullish breakout: price above Camarilla R3 + above 1d EMA34
                if curr_breakout_up and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below Camarilla S3 + below 1d EMA34
                elif curr_breakout_down and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price closes below Camarilla S3 (reversal) or above Camarilla R4 (take profit)
            camarilla_r4 = prev_close[i] + ((prev_high[i] - prev_low[i]) * 1.1 / 2)  # R4
            camarilla_s3 = camarilla_low[i]  # S3
            if curr_close < camarilla_s3 or curr_close > camarilla_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla R3 (reversal) or below Camarilla S4 (take profit)
            camarilla_r3 = camarilla_high[i]  # R3
            camarilla_s4 = prev_close[i] - ((prev_high[i] - prev_low[i]) * 1.1 / 2)  # S4
            if curr_close > camarilla_r3 or curr_close < camarilla_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals