#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX trend filter
# - Enter long when 4h price breaks above Donchian(20) high AND 1d volume > 2.0x 20-period volume SMA AND 1d ADX(14) > 25
# - Enter short when 4h price breaks below Donchian(20) low AND 1d volume > 2.0x 20-period volume SMA AND 1d ADX(14) > 25
# - Exit: price returns to Donchian middle (10-period average of high/low) or opposite band touch
# - Donchian breakout captures structured moves in both bull and bear markets
# - Volume confirmation ensures breakouts have participation
# - ADX filter avoids whipsaws in ranging markets
# - Target: 20-50 trades/year to minimize fee drag while capturing high-probability breakouts

name = "4h_1d_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume and ADX filters (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute Donchian channels for 4h data (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_max_20
    donchian_low = low_min_20
    donchian_middle = (donchian_high + donchian_low) / 2  # Midpoint of channel
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute ADX for 1d data (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original arrays
    
    # Calculate Directional Movement (DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR and DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 1d volume aligned for volume confirmation
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Pre-compute 1d close aligned for trend comparison (optional)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        vol_confirm = volume_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1d ADX > 25 (indicating strong trend)
        adx_filter = adx_1d_aligned[i] > 25
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]  # Cross above upper band
        breakout_down = close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]  # Cross below lower band
        
        # Exit conditions
        exit_long = close[i] < donchian_middle[i]  # Return to middle band
        exit_short = close[i] > donchian_middle[i]  # Return to middle band
        exit_opposite_long = close[i] < donchian_low[i]  # Touch lower band while long
        exit_opposite_short = close[i] > donchian_high[i]  # Touch upper band while short
        
        # Trading logic
        if vol_confirm and adx_filter:
            # Long: Donchian breakout above upper band with volume and trend
            if breakout_up:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Donchian breakout below lower band with volume and trend
            elif breakout_down:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and (exit_long or exit_opposite_long):
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and (exit_short or exit_opposite_short):
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation or weak trend: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals