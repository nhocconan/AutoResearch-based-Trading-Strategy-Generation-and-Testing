#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation.
# Bollinger Band squeeze (BBW < 20th percentile) indicates low volatility primed for breakout.
# Direction determined by 1d ADX > 25 (trending) and price position relative to 1d EMA20.
# Volume confirmation requires breakout bar volume > 2x 20-bar average.
# Works in bull markets (buy breakouts above upper band in uptrend) and bear markets (sell breakdowns below lower band in downtrend).
# Target: 12-37 trades/year on 6h (50-150 total over 4 years). Discrete sizing 0.25 to minimize fee drag.

name = "6h_BollingerSqueeze_1dADXTrend_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 6h data ONCE before loop for Bollinger Bands
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Calculate 6h Bollinger Bands (20, 2)
    close_6h = pd.Series(df_6h['close'].values)
    bb_ma = close_6h.rolling(window=20, min_periods=20).mean()
    bb_std = close_6h.rolling(window=20, min_periods=20).std()
    bb_upper = (bb_ma + 2 * bb_std).values
    bb_lower = (bb_ma - 2 * bb_std).values
    bb_width = ((bb_upper - bb_lower) / bb_ma).values  # normalized BB width
    
    # Calculate 6h BBW percentile rank (lookback 50 periods for regime)
    bb_width_series = pd.Series(bb_width)
    bbw_rank = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Align BBW rank and BB bands to 6h primary timeframe
    bbw_rank_aligned = align_htf_to_ltf(prices, df_6h, bbw_rank)
    bb_upper_aligned = align_htf_to_ltf(prices, df_6h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_6h, bb_lower)
    bb_ma_aligned = align_htf_to_ltf(prices, df_6h, bb_ma.values)
    
    # Load 1d data ONCE before loop for ADX and EMA20 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    dm_plus = high_1d.diff()
    dm_minus = low_1d.diff().multiply(-1)
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).mean()
    
    # DI+ and DI-
    di_plus = 100 * (dm_plus_smooth / atr_1d)
    di_minus = 100 * (dm_minus_smooth / atr_1d)
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # 1d EMA20 for trend direction
    ema_20_1d = close_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 6h primary timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for BBW rank (50) + ADX (14+14) + EMA20 (20)
    
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
        if (np.isnan(bbw_rank_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_bbw_rank = bbw_rank_aligned[i]
        curr_adx = adx_aligned[i]
        curr_ema_20_1d = ema_20_1d_aligned[i]
        curr_bb_upper = bb_upper_aligned[i]
        curr_bb_lower = bb_lower_aligned[i]
        
        # Volume confirmation: current 6h volume > 2x 20-period average
        vol_6h = df_6h['volume'].values
        vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
        vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
        curr_vol_ma = vol_ma_6h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Squeeze condition: BBW rank < 0.2 (bottom 20% = low volatility)
        is_squeeze = curr_bbw_rank < 0.2
        
        # Trend and direction from 1d
        is_trending = curr_adx > 25
        is_uptrend = curr_close > curr_ema_20_1d
        is_downtrend = curr_close < curr_ema_20_1d
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: squeeze breakout above upper band AND uptrend AND volume confirmation
            if (is_squeeze and 
                curr_close > curr_bb_upper and 
                is_trending and 
                is_uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakdown below lower band AND downtrend AND volume confirmation
            elif (is_squeeze and 
                  curr_close < curr_bb_lower and 
                  is_trending and 
                  is_downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price re-enters BB bands (mean reversion) OR trend weakens (ADX < 20)
            if (curr_close < curr_bb_ma_aligned[i] or 
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price re-enters BB bands (mean reversion) OR trend weakens (ADX < 20)
            if (curr_close > curr_bb_ma_aligned[i] or 
                curr_adx < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals