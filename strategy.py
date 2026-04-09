#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d ADX trend filter
# - Primary signal: Price breaks above/below Camarilla H3/L3 levels on 1h using prior 1h session's OHLC
# - Volume confirmation: 4h volume > 1.3x 20-period average volume (avoid low-participation breakouts)
# - Trend filter: 1d ADX > 25 ensures we only trade in trending markets (avoid chop)
# - Works in bull/bear: ADX filter ensures we capture strong moves; volume confirms participation
# - Position size: 0.20 discrete level to minimize fee churn
# - Target: 15-30 trades/year (60-120 total over 4 years) for 1h strategy
# - Session filter: 08-20 UTC to avoid low-liquidity Asian session
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) on 1h

name = "1h_4h_1d_camarilla_breakout_volume_adx_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h volume spike filter
    volume_4h = df_4h['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (1.3 * avg_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    # Handle first period
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 1h ATR(14) for stoploss
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    tr_1h1 = high_1h - low_1h
    tr_1h2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr_1h3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr_1h = np.maximum(tr_1h1, np.maximum(tr_1h2, tr_1h3))
    tr_1h[0] = tr_1h1[0]
    atr_14 = wilders_smoothing(tr_1h, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (not session_filter[i] or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 1h bar using prior bar's OHLC
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Prior 1h bar's OHLC (for Camarilla calculation)
        prev_high = high_1h[i-1]
        prev_low = low_1h[i-1]
        prev_close = close_1h[i-1]
        
        # Camarilla levels (based on prior bar's range)
        range_val = prev_high - prev_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla H3 and L3 levels
        camarilla_h3 = prev_close + (range_val * 1.1 / 4)
        camarilla_l3 = prev_close - (range_val * 1.1 / 4)
        
        if position == 1:  # Long position
            # Exit: price drops below L3 OR stoploss hit
            if close_1h[i] < camarilla_l3 or close_1h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price rises above H3 OR stoploss hit
            if close_1h[i] > camarilla_h3 or close_1h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakouts with volume spike and ADX trend filter
            # Only trade when ADX > 25 (trending market)
            if volume_spike_aligned[i] and adx_aligned[i] > 25:
                # Long: price breaks above H3
                if close_1h[i] > camarilla_h3:
                    position = 1
                    entry_price = close_1h[i]
                    signals[i] = 0.20
                # Short: price breaks below L3
                elif close_1h[i] < camarilla_l3:
                    position = -1
                    entry_price = close_1h[i]
                    signals[i] = -0.20
    
    return signals