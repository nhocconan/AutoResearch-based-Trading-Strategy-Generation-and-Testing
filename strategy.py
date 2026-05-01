#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX regime + volume confirmation
# Williams %R(14) identifies overbought/oversold conditions for mean reversion entries
# 1d ADX(14) > 25 filters for trending markets (avoid chop) while < 20 filters for ranging markets
# In trending markets (ADX>25): fade extreme %R readings (counter-trend)
# In ranging markets (ADX<20): fade %R extremes at support/resistance
# Volume confirmation > 1.5x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~12-37 trades/year per symbol with 0.25 sizing
# Works in both bull and bear markets by adapting to regime via ADX

name = "6h_WilliamsR_1dADX_Regime_Volume_v1"
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
    
    # 1d HTF data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilders_smooth(tr, period)
    dm_plus_smooth = wilders_smooth(dm_plus, period)
    dm_minus_smooth = wilders_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smooth != 0, (dm_plus_smooth / tr_smooth) * 100, 0)
    di_minus = np.where(tr_smooth != 0, (dm_minus_smooth / tr_smooth) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smooth(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R(14) on 6h data
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.where((highest_high - lowest_low) != 0, 
                      -(highest_high - close) / (highest_high - lowest_low) * 100, -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d ADX (30 days) + 6h Williams %R (14 periods)
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(wr[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        wr_val = wr[i]
        
        # Regime filters
        trending_market = adx_val > 25
        ranging_market = adx_val < 20
        
        if position == 0:  # Flat - look for new entries
            if trending_market:
                # In trending markets: fade extreme Williams %R (counter-trend)
                if wr_val < -80 and volume_spike[i]:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif wr_val > -20 and volume_spike[i]:  # Overbought
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif ranging_market:
                # In ranging markets: fade %R extremes at support/resistance
                if wr_val < -80 and volume_spike[i]:  # Oversold -> long
                    signals[i] = 0.25
                    position = 1
                elif wr_val > -20 and volume_spike[i]:  # Overbought -> short
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime (ADX 20-25): no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R returns to neutral territory (-50) or adverse extreme
            if wr_val > -50:  # Return to neutral
                signals[i] = 0.0
                position = 0
            elif wr_val > -10:  # Extreme overbought - reverse
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R returns to neutral territory (-50) or adverse extreme
            if wr_val < -50:  # Return to neutral
                signals[i] = 0.0
                position = 0
            elif wr_val < -90:  # Extreme oversold - reverse
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals