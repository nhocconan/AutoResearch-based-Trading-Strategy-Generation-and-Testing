#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX regime filter and volume confirmation
# Williams %R identifies overbought/oversold conditions: long when %R < -80 (oversold), short when %R > -20 (overbought)
# 1d ADX > 25 indicates trending market (follow Williams %R signals), ADX < 20 indicates ranging (fade extremes)
# Volume confirmation > 1.3x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~10-20 trades/year per symbol with 0.25 sizing
# Works in bull/bear markets by adapting to 1d ADX regime: trend follow in strong trends, mean revert in ranging

name = "6h_WilliamsR_1dADX_Regime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth with Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Wilder smoothing: today = (yesterday * (period-1) + today) / period
            for i in range(period, len(data)):
                if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h Williams %R (14-period)
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * (highest_high - close) / (highest_high - lowest_low), -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: volume > 1.3 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, 20)  # Need 1d ADX, Williams %R, volume EMA
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(wr[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        wr_val = wr[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: ADX > 25 = trending (follow signals), ADX < 20 = ranging (fade extremes)
        if adx_val > 25:  # Trending market - follow Williams %R signals
            if position == 0:  # Flat - look for new entries
                if wr_val < -80 and vol_spike:  # Oversold + volume spike = long
                    signals[i] = 0.25
                    position = 1
                elif wr_val > -20 and vol_spike:  # Overbought + volume spike = short
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:  # Long position
                # Exit: Williams %R returns above -50 (momentum fading)
                if wr_val > -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit: Williams %R returns below -50 (momentum fading)
                if wr_val < -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:  # Ranging market (ADX < 25) - fade extremes
            if position == 0:  # Flat - look for mean reversion entries
                if wr_val < -80 and vol_spike:  # Deep oversold = long (mean revert)
                    signals[i] = 0.25
                    position = 1
                elif wr_val > -20 and vol_spike:  # Deep overbought = short (mean revert)
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:  # Long position
                # Exit: Williams %R returns above -80 (recovered from oversold)
                if wr_val > -80:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit: Williams %R returns below -20 (recovered from overbought)
                if wr_val < -20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals