#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and 1w ADX > 25 regime filter
# Uses primary timeframe 1d for lower trade frequency (~15-25 trades/year) to minimize fee drag
# Donchian breakout provides clear trend structure with defined risk parameters
# 1w volume confirmation ensures institutional participation (reduces false breakouts)
# 1w ADX > 25 filters for trending markets only, avoiding whipsaws in ranging conditions
# Designed to work in both bull and bear markets: ADX filter adapts to regime, volume confirms conviction
# Discrete position sizing (0.25) balances return potential with drawdown control

name = "1d_Donchian20_1wVolume_1wADX_Regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels (primary timeframe calculation)
    # Donchian(20): upper = 20-period high, lower = 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1w HTF data for volume and regime filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    # 1w volume spike filter: volume > 2.0 * 20-period EMA (strict for quality)
    vol_1w = df_1w['volume'].values
    vol_series_1w = pd.Series(vol_1w)
    vol_ema_20_1w = vol_series_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1w = vol_1w > (2.0 * vol_ema_20_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    # 1w ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing function
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            first_val = np.nansum(x[1:period+1])
            result[period] = first_val
            for i in range(period+1, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    # Calculate ADX components
    tr_period = 14
    tr_smoothed = wilders_smoothing(tr, tr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, tr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, tr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, tr_period)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period for all indicators
    start_idx = max(34, 20)  # ADX needs ~34 bars (20 for Donchian + 14 for ADX smoothing)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in strongly trending markets (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            if strong_trend:
                # Long: Break above Donchian upper with 1w volume confirmation
                if close[i] > donchian_upper[i] and volume_spike_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Break below Donchian lower with 1w volume confirmation
                elif close[i] < donchian_lower[i] and volume_spike_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid weak/ranging markets
        
        elif position == 1:  # Long position
            # Exit: price returns to Donchian lower or opposite breakout with volume
            if close[i] <= donchian_lower[i] or (close[i] < donchian_lower[i] and volume_spike_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to Donchian upper or opposite breakout with volume
            if close[i] >= donchian_upper[i] or (close[i] > donchian_upper[i] and volume_spike_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals