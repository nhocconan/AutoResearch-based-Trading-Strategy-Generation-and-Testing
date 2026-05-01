#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation.
# Long when: Bollinger Bands squeeze (BBW < 20th percentile) AND price breaks above upper band AND 1d ADX > 25 AND 6h volume > 1.5x 20-period average
# Short when: Bollinger Bands squeeze (BBW < 20th percentile) AND price breaks below lower band AND 1d ADX > 25 AND 6h volume > 1.5x 20-period average
# Uses discrete sizing 0.25. Target: 12-37 trades/year on 6h.
# Bollinger squeeze identifies low volatility primed for breakout, 1d ADX ensures higher timeframe trend strength, volume confirms breakout conviction.
# Works in bull (catching upside breakouts) and bear (catching downside breakdowns) by trading breakouts in the direction of the 1d trend.

name = "6h_BB_Squeeze_ADX_VolumeBreakout_v1"
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
    
    # Load 6h data ONCE before loop for Bollinger Bands and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Bollinger Bands on 6h (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_band = (sma_20 + 2 * std_20).values
    lower_band = (sma_20 - 2 * std_20).values
    bb_width = ((upper_band - lower_band) / sma_20.values) * 100  # BBW as percentage
    
    # Percentile of BBW (20-period lookback for squeeze condition)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first period has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = dm_minus[0] = 0  # first period
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to 6h primary timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_6h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_6h, lower_band)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_6h, bb_width_percentile)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h volume average (20-period) for volume confirmation
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Bollinger Bands and volume MA
    
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
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma_6h_aligned[i]
        curr_upper = upper_band_aligned[i]
        curr_lower = lower_band_aligned[i]
        curr_bb_percentile = bb_width_percentile_aligned[i]
        curr_adx = adx_aligned[i]
        
        # Bollinger Band squeeze: BBW < 20th percentile (low volatility)
        squeeze = curr_bb_percentile < 20.0
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # 1d trend filter: ADX > 25 indicates strong trend
        strong_trend = curr_adx > 25.0
        
        # Breakout conditions
        bullish_breakout = curr_close > curr_upper
        bearish_breakout = curr_close < curr_lower
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: squeeze AND bullish breakout AND strong trend AND volume confirm
            if (squeeze and 
                bullish_breakout and 
                strong_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: squeeze AND bearish breakout AND strong trend AND volume confirm
            elif (squeeze and 
                  bearish_breakout and 
                  strong_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price closes below middle Bollinger Band (SMA20) OR loses volume confirmation
            sma_20_val = pd.Series(close).rolling(window=20, min_periods=20).mean().values[i]
            if (curr_close < sma_20_val or 
                not volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above middle Bollinger Band (SMA20) OR loses volume confirmation
            sma_20_val = pd.Series(close).rolling(window=20, min_periods=20).mean().values[i]
            if (curr_close > sma_20_val or 
                not volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals