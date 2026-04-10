#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and 1w ADX trend filter
# - Long when price breaks above Donchian upper channel (20-period) with volume > 1.3x 20-period EMA and 1w ADX > 20
# - Short when price breaks below Donchian lower channel (20-period) with volume > 1.3x 20-period EMA and 1w ADX > 20
# - Exit: ATR trailing stop (1.5x ATR) or Donchian channel reversion (opposite side)
# - Position sizing: 0.25 discrete level
# - Targets ~20-30 trades/year on 4h timeframe. Donchian channels provide clear structure,
#   volume confirmation validates breakout strength, ADX filter avoids choppy markets.
#   Works in bull/bear: breakouts work in both regimes, ADX filter avoids false signals in ranging markets.

name = "4h_1d_1w_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Calculate Donchian channels (20-period) from previous 1d
    # Based on previous day's high, low, close
    prev_high_1d = df_1d['high'].shift(1).values  # Previous day's high
    prev_low_1d = df_1d['low'].shift(1).values    # Previous day's low
    prev_close_1d = df_1d['close'].shift(1).values # Previous day's close
    
    # Donchian channels: upper = max(high, prev_close), lower = min(low, prev_close)
    donchian_upper = np.maximum(prev_high_1d, prev_close_1d)
    donchian_lower = np.minimum(prev_low_1d, prev_close_1d)
    
    # Align Donchian levels to 4h timeframe (using previous day's levels)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 1d volume EMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_ema_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ema_20_1d)
    
    # Calculate 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / np.where(tr_14 == 0, 1e-10, tr_14)
    di_minus = 100 * dm_minus_14 / np.where(tr_14 == 0, 1e-10, tr_14)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1e-10, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate ATR(14) for trailing stop on 4h
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_ema_20_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.3 * volume_ema_20_1d_aligned[i]
        
        # Trend filter: 1w ADX > 20 indicates trending market
        trend_filter = adx_aligned[i] > 20
        
        # Donchian breakout entry conditions
        # Long: price breaks above Donchian upper channel
        # Short: price breaks below Donchian lower channel
        long_entry = (close[i] > donchian_upper_aligned[i] and 
                     vol_confirm and 
                     trend_filter)
        short_entry = (close[i] < donchian_lower_aligned[i] and 
                      vol_confirm and 
                      trend_filter)
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif short_entry:
                position = -1
                signals[i] = -0.25
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Update highest/lowest since entry
            if position == 1:
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price drops 1.5*ATR from high
                # or Donchian lower channel reversion (mean reversion)
                if (close[i] < highest_since_entry - 1.5 * atr_4h[i] or  # trailing stop
                    close[i] < donchian_lower_aligned[i]):         # Donchian lower reversion
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price rises 1.5*ATR from low
                # or Donchian upper channel reversion (mean reversion)
                if (close[i] > lowest_since_entry + 1.5 * atr_4h[i] or  # trailing stop
                    close[i] > donchian_upper_aligned[i]):         # Donchian upper reversion
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals