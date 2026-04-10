#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ADX trend filter
# - Long when price breaks above 20-period Donchian high with volume > 1.8x 20-bar average AND 1d ADX > 25 (trending)
# - Short when price breaks below 20-period Donchian low with volume > 1.8x 20-bar average AND 1d ADX > 25 (trending)
# - Exit when price retreats to midpoint of Donchian channel OR volume drops below 0.7x average
# - Uses 1d ADX to filter for trending markets only, avoiding whipsaws in ranging conditions
# - Tight entry conditions targeting 15-25 trades/year (60-100 total over 4 years)
# - Donchian breakouts capture strong momentum moves which work well in both bull and bear regimes when combined with trend filter

name = "12h_1d_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.7x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.7 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute aligned 1d data properly
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Align them to 12h timeframe
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d ADX(14) for trend filter
    # Calculate True Range
    tr1 = np.abs(h_1d[1:] - l_1d[1:])
    tr2 = np.abs(h_1d[1:] - c_1d[:-1])
    tr3 = np.abs(l_1d[1:] - c_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate Directional Movement
    up_move = h_1d[1:] - h_1d[:-1]
    down_move = l_1d[:-1] - l_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR and DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 1d bar values (need to shift by 2 to avoid look-ahead)
        # Since 12h timeframe, there are 2 bars per 1d bar
        if i >= 4:  # Need at least 4 12h bars (2x 1d bars) to get previous 1d bar's data
            # Get index of previous completed 1d bar
            prev_1d_idx = i - 2  # Look back 2 bars (one 12h period)
            
            if prev_1d_idx >= 0 and not (np.isnan(h_1d_aligned[prev_1d_idx]) or 
                                        np.isnan(l_1d_aligned[prev_1d_idx]) or 
                                        np.isnan(c_1d_aligned[prev_1d_idx])):
                ph = h_1d_aligned[prev_1d_idx]  # Previous 12h period's high
                pl = l_1d_aligned[prev_1d_idx]  # Previous 12h period's low
                pc = c_1d_aligned[prev_1d_idx]  # Previous 12h period's close
                
                # Calculate Donchian levels (20-period)
                if i >= 40:  # Need 20 periods of 12h data for Donchian (20 * 2 = 40 12h bars)
                    # Look back 20 periods of 12h data (40 12h bars) for Donchian calculation
                    donchian_start = i - 40
                    donchian_end = i - 2  # Previous completed 12h bar
                    
                    if donchian_start >= 0:
                        # Get high and low for the lookback period
                        lookback_highs = h_1d_aligned[donchian_start:donchian_end+1]
                        lookback_lows = l_1d_aligned[donchian_start:donchian_end+1]
                        
                        # Filter out NaN values
                        valid_highs = lookback_highs[~np.isnan(lookback_highs)]
                        valid_lows = lookback_lows[~np.isnan(lookback_lows)]
                        
                        if len(valid_highs) > 0 and len(valid_lows) > 0:
                            upper_channel = np.max(valid_highs)
                            lower_channel = np.min(valid_lows)
                            channel_mid = (upper_channel + lower_channel) / 2
                            
                            if position == 0:  # Flat - look for new breakout entries
                                # Long breakout: price > Donchian upper channel with volume spike AND 1d ADX > 25
                                if (prices['close'].iloc[i] > upper_channel and 
                                    vol_spike.iloc[i] and 
                                    adx_aligned[i] > 25):
                                    position = 1
                                    signals[i] = 0.25
                                # Short breakdown: price < Donchian lower channel with volume spike AND 1d ADX > 25
                                elif (prices['close'].iloc[i] < lower_channel and 
                                      vol_spike.iloc[i] and 
                                      adx_aligned[i] > 25):
                                    position = -1
                                    signals[i] = -0.25
                            else:  # Have position - look for exit
                                # Exit conditions:
                                # 1. Price retreats to midpoint of Donchian channel
                                # 2. Volume drops below 0.7x average (loss of momentum)
                                if position == 1:  # Long position
                                    if (prices['close'].iloc[i] < channel_mid or 
                                        vol_weak.iloc[i]):
                                        position = 0
                                        signals[i] = 0.0
                                    else:
                                        signals[i] = 0.25  # Hold long
                                elif position == -1:  # Short position
                                    if (prices['close'].iloc[i] > channel_mid or 
                                        vol_weak.iloc[i]):
                                        position = 0
                                        signals[i] = 0.0
                                    else:
                                        signals[i] = -0.25  # Hold short
                        else:
                            # Hold current position
                            if position == 0:
                                signals[i] = 0.0
                            elif position == 1:
                                signals[i] = 0.25
                            else:
                                signals[i] = -0.25
                    else:
                        # Hold current position
                        if position == 0:
                            signals[i] = 0.0
                        elif position == 1:
                            signals[i] = 0.25
                        else:
                            signals[i] = -0.25
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals