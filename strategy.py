#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike (>1.5x average)
# Camarilla pivots identify key intraday support/resistance levels. Breakouts above R3 or below S3
# with volume confirmation and 4h trend alignment capture strong momentum moves.
# Uses 1h timeframe for precise entry timing, 4h for trend direction.
# Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h Camarilla levels (based on previous bar's OHLC)
    # Pivot = (H + L + C) / 3
    # R3 = H + 2*(Pivot - L)
    # S3 = L - 2*(H - Pivot)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r3 = prev_high + 2 * (pivot - prev_low)
    camarilla_s3 = prev_low - 2 * (prev_high - pivot)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # warmup for volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_ema_34_4h = ema_34_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with trend filter and Camarilla breakout
            if curr_volume_spike:
                # Bullish: price breaks above Camarilla R3 + price above 4h EMA34
                if curr_high > camarilla_r3[i] and curr_close > curr_ema_34_4h:
                    signals[i] = 0.20
                    position = 1
                # Bearish: price breaks below Camarilla S3 + price below 4h EMA34
                elif curr_low < camarilla_s3[i] and curr_close < curr_ema_34_4h:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 (trend reversal)
            if curr_low < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 (trend reversal)
            if curr_high > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals