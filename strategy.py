#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and ADX trend filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 2.0x 20-period average AND 1d ADX(14) > 25
# - Short when price breaks below Camarilla L3 level AND 1d volume > 2.0x 20-period average AND 1d ADX(14) > 25
# - Exit when price returns to Camarilla Pivot point (mid-level)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla pivots provide mathematically derived support/resistance levels that work well in ranging markets
# - Volume confirmation ensures breakout validity
# - ADX filter ensures we only trade in trending conditions to avoid false breakouts in chop
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_camarilla_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, and TR
    tr_period = 14
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    tr_smooth = np.zeros_like(tr)
    
    # Initial values
    plus_dm_smooth[13] = np.mean(plus_dm[1:14])
    minus_dm_smooth[13] = np.mean(minus_dm[1:14])
    tr_smooth[13] = np.mean(tr[1:14])
    
    # Wilder's smoothing
    for i in range(14, len(tr)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
        tr_smooth[i] = (tr_smooth[i-1] * 13 + tr[i]) / 14
    
    # Calculate +DI and -DI
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx_1d = np.zeros_like(dx)
    adx_1d[27] = np.mean(dx[14:28])  # First ADX value
    for i in range(28, len(dx)):
        adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # ADX trend filter: trending when ADX > 25
    trend_filter = adx_1d > 25
    
    # Align HTF indicators to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    trend_filter_aligned = align_htf_to_ltf(prices, df_1d, trend_filter)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(vol_ma[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(trend_filter_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Calculate Camarilla pivot levels for current 12h bar
            # Using previous 12h bar's OHLC
            if i >= 1:
                prev_close = close[i-1]
                prev_high = high[i-1]
                prev_low = low[i-1]
                
                pivot = (prev_high + prev_low + prev_close) / 3
                range_val = prev_high - prev_low
                
                # Camarilla levels
                h3 = pivot + (range_val * 1.1 / 4)
                l3 = pivot - (range_val * 1.1 / 4)
                h4 = pivot + (range_val * 1.1 / 2)
                l4 = pivot - (range_val * 1.1 / 2)
                
                # Long conditions: price breaks above H3 AND volume spike AND trend filter
                if (close[i] > h3 and 
                    volume_spike_aligned[i] and 
                    trend_filter_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short conditions: price breaks below L3 AND volume spike AND trend filter
                elif (close[i] < l3 and 
                      volume_spike_aligned[i] and 
                      trend_filter_aligned[i]):
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Calculate Camarilla pivot for exit condition (return to pivot point)
            if i >= 1:
                prev_close = close[i-1]
                prev_high = high[i-1]
                prev_low = low[i-1]
                
                pivot = (prev_high + prev_low + prev_close) / 3
                
                # Exit conditions: price returns to pivot point
                exit_long = (position == 1 and close[i] <= pivot)
                exit_short = (position == -1 and close[i] >= pivot)
                
                if exit_long or exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    if position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals