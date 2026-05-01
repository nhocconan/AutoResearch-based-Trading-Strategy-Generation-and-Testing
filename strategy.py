#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND 1w ADX > 25 AND volume > 2.0x 24-bar average.
# Short when price breaks below Donchian lower band AND 1w ADX > 25 AND volume > 2.0x 24-bar average.
# Uses discrete sizing 0.25 to balance return and drawdown. No session filter to maximize opportunities.
# Target: 50-150 total trades over 4 years (12-37/year) for BTC/ETH/SOL.
# 1w ADX ensures we only trade strong trends, reducing whipsaw in ranging markets.
# Volume confirmation ensures breakouts have conviction.

name = "6h_Donchian20_1wADX_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1w ADX calculation (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize smoothed values
    atr = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    # First TR is average of first 'period' TR values
    if len(tr) >= period + 1:
        atr[period] = np.nanmean(tr[1:period+1])
        plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
        
        # Wilder's smoothing for subsequent values
        for i in range(period + 1, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
    
    # Calculate +DI and -DI
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    mask = ~np.isnan(atr) & (atr != 0)
    plus_di[mask] = (plus_dm_smooth[mask] / atr[mask]) * 100
    minus_di[mask] = (minus_dm_smooth[mask] / atr[mask]) * 100
    
    # Calculate DX and ADX
    dx = np.full_like(tr, np.nan)
    mask_di = (~np.isnan(plus_di)) & (~np.isnan(minus_di)) & ((plus_di + minus_di) != 0)
    dx[mask_di] = (np.abs(plus_di[mask_di] - minus_di[mask_di]) / (plus_di[mask_di] + minus_di[mask_di])) * 100
    
    # ADX is smoothed DX
    adx = np.full_like(dx, np.nan)
    if len(dx) >= 2 * period:
        # First ADX is average of first 'period' DX values after DX is valid
        valid_dx_start = period  # DX starts being valid after period+1, but we use period for smoothing start
        if not np.isnan(dx[valid_dx_start:valid_dx_start+period]).all():
            adx[valid_dx_start+period-1] = np.nanmean(dx[valid_dx_start:valid_dx_start+period])
            for i in range(valid_dx_start+period, len(dx)):
                if not np.isnan(dx[i]):
                    adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # Align 1w ADX to 6h timeframe (wait for completed weekly bar)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Donchian channels (20-period) on 6h data
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current 6h volume > 2.0x 24-bar average (48h lookback)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup periods for Donchian, volume MA, and ADX
    start_idx = max(lookback, 24, 30) + 5  # Extra buffer for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any required values are NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        adx_strong = adx_aligned[i] > 25.0  # Strong trend filter
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above upper band
        breakout_down = curr_low < donchian_low[i]  # break below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper band AND strong trend AND volume confirmation
            if (breakout_up and 
                adx_strong and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band AND strong trend AND volume confirmation
            elif (breakout_down and 
                  adx_strong and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower band (stoploss) OR ADX weakens (<20)
            if (curr_low < donchian_low[i] or 
                adx_aligned[i] < 20.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (stoploss) OR ADX weakens (<20)
            if (curr_high > donchian_high[i] or 
                adx_aligned[i] < 20.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals