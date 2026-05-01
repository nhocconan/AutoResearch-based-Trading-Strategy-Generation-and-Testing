#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout + 1d Volume Spike + ADX Regime Filter
# Uses Bollinger Band width to detect low volatility squeeze (potential breakout setup).
# Enters long/short on BB breakout in direction of 1d ADX trend (ADX > 25) with volume confirmation.
# Volume spike > 1.5x 20-bar average confirms institutional participation.
# Works in bull markets (trend continuation) and bear markets (trend continuation down).
# Target: 12-25 trades/year by requiring squeeze, breakout, volume, and trend alignment.

name = "6h_BBSqueeze_Breakout_ADXTrend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for ADX and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (trend strength filter)
    # ADX requires +DI, -DI, and TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        """Wilder's smoothing (similar to EMA with alpha=1/period)"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        smoothed = np.full_like(values, np.nan)
        smoothed[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            if not np.isnan(smoothed[i-1]):
                smoothed[i] = (smoothed[i-1] * (period-1) + values[i]) / period
            else:
                smoothed[i] = np.nan
        return smoothed
    
    period_adx = 14
    atr = wilders_smoothing(tr, period_adx)
    plus_di = 100 * wilders_smoothing(plus_dm, period_adx) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period_adx) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period_adx)
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d volume average for spike detection
    vol_20_avg = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_20_avg)
    
    # Bollinger Bands on 6h close (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_s.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_mid + (bb_std_dev * bb_std)
    bb_lower = bb_mid - (bb_std_dev * bb_std)
    bb_width = (bb_upper - bb_lower) / bb_mid  # normalized width
    
    # Bollinger Band Squeeze: width < 20th percentile of last 100 bars
    def rolling_percentile(values, window, percentile):
        """Rolling percentile calculation"""
        if len(values) < window:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        for i in range(window-1, len(values)):
            window_data = values[i-window+1:i+1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                result[i] = np.percentile(valid_data, percentile)
            else:
                result[i] = np.nan
        return result
    
    bb_width_20th = rolling_percentile(bb_width, 100, 20.0)
    squeeze = bb_width < bb_width_20th
    
    # Volume spike on 6h: current volume > 1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 30)  # warmup for BB and ADX
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_20_avg_aligned[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(bb_mid[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_adx = adx_aligned[i]
        curr_vol_20_avg = vol_20_avg_aligned[i]
        curr_bb_upper = bb_upper[i]
        curr_bb_lower = bb_lower[i]
        curr_bb_mid = bb_mid[i]
        curr_squeeze = squeeze[i]
        curr_volume_spike = volume_spike[i]
        
        # Trend condition: ADX > 25 indicates strong trend
        strong_trend = curr_adx > 25
        
        # Determine trend direction from +DI/-DI (need 1d values for direction)
        # We'll approximate using price action: if close > BB mid, bullish bias; else bearish
        # More robust: use 1d close vs prior close for direction
        if i >= start_idx + 1:
            # Get prior 1d close direction (approximate)
            prev_close_6h = close[i-1]
            curr_bullish_bias = curr_close > curr_bb_mid
            curr_bearish_bias = curr_close < curr_bb_mid
        else:
            curr_bullish_bias = False
            curr_bearish_bias = False
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze breakout above upper band + volume spike + strong trend + bullish bias
            if (curr_squeeze and 
                curr_high > curr_bb_upper and  # breakout confirmation
                curr_volume_spike and
                strong_trend and
                curr_bullish_bias):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower band + volume spike + strong trend + bearish bias
            elif (curr_squeeze and
                  curr_low < curr_bb_lower and  # breakout confirmation
                  curr_volume_spike and
                  strong_trend and
                  curr_bearish_bias):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below BB middle OR volatility expands (squeeze ends) OR adverse move
            if (curr_close < curr_bb_mid or 
                not curr_squeeze or  # volatility expansion
                curr_low < curr_bb_lower * 0.99):  # stop if breaks below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above BB middle OR volatility expands OR adverse move
            if (curr_close > curr_bb_mid or 
                not curr_squeeze or  # volatility expansion
                curr_high > bb_upper[i] * 1.01):  # stop if breaks above upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals