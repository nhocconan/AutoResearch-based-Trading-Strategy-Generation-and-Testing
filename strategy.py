#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1w ADX trend filter
# - Long when price breaks above Camarilla H3 level with volume > 1.5x 20-period EMA and 1w ADX > 25 (trending)
# - Short when price breaks below Camarilla L3 level with volume > 1.5x 20-period EMA and 1w ADX > 25 (trending)
# - Exit: ATR trailing stop (2.0x ATR) or Camarilla L4/H4 reversion
# - Position sizing: 0.25 discrete level
# - Targets ~20-30 trades/year on 4h timeframe. Camarilla pivots provide mathematical support/resistance,
#   volume confirmation validates institutional participation, ADX filter ensures we only trade in trending markets
#   where breakouts are more likely to succeed. Works in bull/bear: breakouts work in both regimes,
#   ADX filter avoids choppy markets where false breakouts occur.

name = "4h_1d_1w_camarilla_volume_adx_v2"
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
    
    # Calculate Camarilla pivot levels from previous 1d
    # Based on previous day's high, low, close
    prev_high_1d = df_1d['high'].shift(1).values  # Previous day's high
    prev_low_1d = df_1d['low'].shift(1).values    # Previous day's low
    prev_close_1d = df_1d['close'].shift(1).values # Previous day's close
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.125*(high-low)
    # H2 = close + 0.75*(high-low)
    # H1 = close + 0.5*(high-low)
    # L1 = close - 0.5*(high-low)
    # L2 = close - 0.75*(high-low)
    # L3 = close - 1.125*(high-low)
    # L4 = close - 1.5*(high-low)
    camarilla_h4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_h3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_h2 = prev_close_1d + 0.75 * (prev_high_1d - prev_low_1d)
    camarilla_h1 = prev_close_1d + 0.5 * (prev_high_1d - prev_low_1d)
    camarilla_l1 = prev_close_1d - 0.5 * (prev_high_1d - prev_low_1d)
    camarilla_l2 = prev_close_1d - 0.75 * (prev_high_1d - prev_low_1d)
    camarilla_l3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_l4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
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
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ema_20_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_ema_20_1d_aligned[i]
        
        # Trend filter: 1w ADX > 25 indicates trending market
        trend_filter = adx_aligned[i] > 25
        
        # Camarilla breakout entry conditions
        # Long: price breaks above Camarilla H3 level
        # Short: price breaks below Camarilla L3 level
        long_entry = (close[i] > camarilla_h3_aligned[i] and 
                     vol_confirm and 
                     trend_filter)
        short_entry = (close[i] < camarilla_l3_aligned[i] and 
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
                # ATR trailing stop: exit if price drops 2.0*ATR from high
                # or Camarilla L4 reversion (mean reversion)
                if (close[i] < highest_since_entry - 2.0 * atr_4h[i] or  # trailing stop
                    close[i] < camarilla_l4_aligned[i]):         # Camarilla L4 reversion
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                highest_since_entry = max(highest_since_entry, close[i])
                lowest_since_entry = min(lowest_since_entry, close[i])
                # ATR trailing stop: exit if price rises 2.0*ATR from low
                # or Camarilla H4 reversion (mean reversion)
                if (close[i] > lowest_since_entry + 2.0 * atr_4h[i] or  # trailing stop
                    close[i] > camarilla_h4_aligned[i]):         # Camarilla H4 reversion
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals