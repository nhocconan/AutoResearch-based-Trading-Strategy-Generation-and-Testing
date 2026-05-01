#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d ADX regime filter and volume spike
# Williams %R identifies overbought/oversold conditions. In ranging markets (ADX<25),
# we fade extremes: long when %R<-80 and volume spike, short when %R>-20 and volume spike.
# In trending markets (ADX>25), we breakout: long when %R>-20 and volume spike, short when %R<-80 and volume spike.
# This adaptive approach works in both bull and bear markets by switching between mean reversion and breakout.
# Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag

name = "6h_WilliamsR_Extreme_Reversal_1dADX_Regime_VolumeSpike_v1"
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
    
    # 1d ADX calculation (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = np.diff(high_1d)
    down_move = -np.diff(low_1d)  # Negative because we want positive values for down moves
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe (with 1-bar delay for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R (14-period) on 6h data
    def williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    wr = williams_r(high, low, close, 14)
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(30, 20, 14)  # Need sufficient history for 1d ADX, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(wr[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 25 = ranging
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 25
        
        # Williams %R extremes
        wr_overbought = wr[i] > -20   # Overbought
        wr_oversold = wr[i] < -80     # Oversold
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            if is_ranging:
                # In ranging markets: fade extremes (mean reversion)
                if wr_oversold and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif wr_overbought and vol_spike:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # is_trending
                # In trending markets: breakout with momentum
                if wr_oversold and vol_spike:
                    signals[i] = 0.25
                    position = 1
                elif wr_overbought and vol_spike:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            if is_ranging:
                # In ranging markets: exit when WR returns to neutral or overbought
                if wr[i] > -50:  # Return to neutral or overbought
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # is_trending
                # In trending markets: exit when WR shows weakness or reversal
                if wr[i] < -80:  # Oversold again (potential reversal)
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            if is_ranging:
                # In ranging markets: exit when WR returns to neutral or oversold
                if wr[i] < -50:  # Return to neutral or oversold
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # is_trending
                # In trending markets: exit when WR shows strength or reversal
                if wr[i] > -20:  # Overbought again (potential reversal)
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals