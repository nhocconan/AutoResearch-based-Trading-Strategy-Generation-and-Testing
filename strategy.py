#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3, close > 1d EMA34, and volume > 2.0x 24-bar avg.
# Short when price breaks below S3, close < 1d EMA34, and volume > 2.0x 24-bar avg.
# Exit when price re-enters the Camarilla H-L range (between H3 and L3).
# Uses 12h timeframe for lower trade frequency (target: 12-37 trades/year) to minimize fee drag.
# Camarilla levels provide precise intraday support/resistance based on prior day's range.
# 1d EMA34 filters for higher timeframe trend alignment.
# Volume confirmation with moderate threshold reduces false breakouts.
# Works in bull markets via breakouts with trend and in bear markets via breakdowns with trend.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 12h bar (HLC of previous bar)
    # Camarilla: H4 = (H-L)*1.1/2 + C, L4 = C - (H-L)*1.1/2
    # R3 = H4, S3 = L4
    # We use the prior completed 12h bar's HLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    hl_range = prev_high - prev_low
    camarilla_h4 = prev_close + (hl_range * 1.1 / 2)
    camarilla_l4 = prev_close - (hl_range * 1.1 / 2)
    
    # Volume confirmation: volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for EMA34 and Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_h4 = camarilla_h4[i]
        curr_l4 = camarilla_l4[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above H4 (R3), close > 1d EMA34, volume spike
            if (curr_close > curr_h4 and 
                curr_close > curr_ema_34_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 (S3), close < 1d EMA34, volume spike
            elif (curr_close < curr_l4 and 
                  curr_close < curr_ema_34_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price re-enters the Camarilla H-L range (below H4)
            if curr_close < curr_h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price re-enters the Camarilla H-L range (above L4)
            if curr_close > curr_l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals