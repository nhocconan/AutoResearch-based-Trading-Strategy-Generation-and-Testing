#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout + 4h ADX trend filter + volume confirmation
# - Primary signal: Break above/below Camarilla H3/L3 levels on 1h (intraday pivot structure)
# - Trend filter: 4h ADX > 25 ensures we only trade in trending markets (avoids chop)
# - Volume confirmation: 1h volume > 20-period EMA of volume (avoids low-participation breakouts)
# - Session filter: 08-20 UTC to focus on liquid London/NY overlap, avoid Asian session noise
# - Position size: 0.20 (discrete level) to minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) for 1h strategy
# - Works in bull/bear: Camarilla levels adapt to volatility, ADX filter avoids false breakouts in ranging markets

name = "1h_4h_camarilla_adx_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h ADX for trend filter (ADX > 25 = trending)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h),
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)),
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[:period-1] = np.nan
        if len(values) >= period:
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)  # ADX is smoothed DX
    
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Pre-compute 1h volume regime: volume > 20-period EMA of volume
    volume = prices['volume'].values
    volume_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_regime = volume > volume_ema_20
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(adx_aligned[i]) or
            np.isnan(volume_regime[i]) or
            not (8 <= hours[i] <= 20)):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for 1h using previous bar's OHLC
        if i >= 1:
            high_prev = prices['high'].iloc[i-1]
            low_prev = prices['low'].iloc[i-1]
            close_prev = prices['close'].iloc[i-1]
            range_prev = high_prev - low_prev
            
            if range_prev > 0:
                camarilla_h3 = close_prev + (range_prev * 1.1 / 4)
                camarilla_l3 = close_prev - (range_prev * 1.1 / 4)
            else:
                camarilla_h3 = high_prev
                camarilla_l3 = low_prev
        else:
            camarilla_h3 = prices['high'].iloc[i]
            camarilla_l3 = prices['low'].iloc[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Camarilla H3 OR ADX drops below 20 (trend weakening)
            if prices['close'].iloc[i] < camarilla_h3 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above Camarilla L3 OR ADX drops below 20 (trend weakening)
            if prices['close'].iloc[i] > camarilla_l3 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakouts with volume confirmation and ADX filter
            # Long: price breaks above Camarilla H3 AND volume regime AND ADX > 25
            if (prices['close'].iloc[i] > camarilla_h3 and 
                volume_regime[i] and 
                adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below Camarilla L3 AND volume regime AND ADX > 25
            elif (prices['close'].iloc[i] < camarilla_l3 and 
                  volume_regime[i] and 
                  adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.20
    
    return signals