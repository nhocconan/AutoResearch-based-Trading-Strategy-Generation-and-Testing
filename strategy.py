#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w ADX trend filter
# - Long when price breaks above Donchian upper band (20) AND 1d volume > 1.3x 20-bar avg AND 1w ADX > 25 (trending)
# - Short when price breaks below Donchian lower band (20) AND 1d volume > 1.3x 20-bar avg AND 1w ADX > 25 (trending)
# - Exit when price touches Donchian middle band (20) or ATR-based stoploss (2*ATR against position)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian provides clear structure, volume confirms breakout strength, ADX filter avoids choppy markets
# - Works in both bull (breakouts continue) and bear (breakdowns continue) markets when ADX confirms trend

name = "4h_1d_1w_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute Donchian channels from 4h data (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower/middle bands (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 1d volume confirmation: > 1.3x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.3 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute 1w ADX trend filter (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range for ADX
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
    tr_w_smooth = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_w_smooth > 0, 100 * dm_plus_smooth / tr_w_smooth, 0)
    di_minus = np.where(tr_w_smooth > 0, 100 * dm_minus_smooth / tr_w_smooth, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d volume spike and 1w ADX to 4h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(atr[i]) or np.isnan(vol_spike_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper band AND volume spike AND ADX > 25 (trending)
            if (prices['high'].iloc[i] > donchian_upper[i] and 
                vol_spike_1d_aligned[i] and 
                adx_aligned[i] > 25):
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = 0.25
            # Short when price breaks below Donchian lower band AND volume spike AND ADX > 25 (trending)
            elif (prices['low'].iloc[i] < donchian_lower[i] and 
                  vol_spike_1d_aligned[i] and 
                  adx_aligned[i] > 25):
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit conditions
            # Exit conditions: return to middle band OR ATR stoploss hit
            exit_signal = False
            if position == 1:  # Long position
                # Exit to middle band (mean reversion)
                if prices['low'].iloc[i] <= donchian_middle[i]:
                    exit_signal = True
                # ATR stoploss (2*ATR against position)
                elif prices['low'].iloc[i] < entry_price - 2.0 * atr[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                # Exit to middle band (mean reversion)
                if prices['high'].iloc[i] >= donchian_middle[i]:
                    exit_signal = True
                # ATR stoploss (2*ATR against position)
                elif prices['high'].iloc[i] > entry_price + 2.0 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals