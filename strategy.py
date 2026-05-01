#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d ADX trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-bar average.
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-bar average.
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 6h timeframe to capture medium-term mean reversion in trending markets.
# Williams %R identifies overextended moves, ADX ensures we only trade in trending conditions where mean reversion works best.
# Volume confirmation reduces false signals. Works in both bull and bear markets as it trades pullbacks within trends.

name = "6h_WilliamsR_1dADX_Trend_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
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
    
    # Williams %R calculation (14-period) on 6h data
    def williams_r(high, low, close, period=14):
        highest_high = np.maximum.accumulate(high)
        lowest_low = np.minimum.accumulate(low)
        # For proper lookback, we need to use rolling window
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = np.where((highest_high - lowest_low) != 0,
                      -100 * (highest_high - close) / (highest_high - lowest_low),
                      -50)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: current 6h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if np.isnan(wr[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if vol_ma[i] <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = volume[i] > (vol_ma[i] * 1.5)
        adx_trending = adx_aligned[i] > 25
        
        # Williams %R levels
        wr_oversold = wr[i] < -80
        wr_overbought = wr[i] > -20
        wr_exit_long = wr[i] > -50  # Exit long when WR crosses above -50
        wr_exit_short = wr[i] < -50  # Exit short when WR crosses below -50
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: oversold AND trending AND volume confirmation
            if (wr_oversold and 
                adx_trending and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought AND trending AND volume confirmation
            elif (wr_overbought and 
                  adx_trending and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: WR crosses above -50 (mean reversion complete) OR ADX loses trend
            if (wr_exit_long or 
                not adx_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: WR crosses below -50 (mean reversion complete) OR ADX loses trend
            if (wr_exit_short or 
                not adx_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals