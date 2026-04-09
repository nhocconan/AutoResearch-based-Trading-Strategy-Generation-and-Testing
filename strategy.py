#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal + 12h volume confirmation + 1d regime filter
# - Entry signal: Williams %R(14) crosses above -80 (oversold) for long, below -20 (overbought) for short on 6h
# - Volume confirmation: 12h volume > 1.3x 20-period average volume (avoid false reversals)
# - Regime filter: 1d ADX(14) < 25 (range market) enables mean reversion trades
# - Works in bull/bear: In ranging markets (ADX < 25), Williams %R reversals are reliable; in trending markets, filter reduces false signals
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 50-150 total trades over 4 years (12-37/year) per 6h strategy guidelines

name = "6h_12h_1d_williams_r_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h volume confirmation
    volume_12h = df_12h['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_12h > (1.3 * avg_volume_20)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm)
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
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
    
    atr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_14 > 0, (dm_plus_14 / atr_14) * 100, 0)
    di_minus = np.where(atr_14 > 0, (dm_minus_14 / atr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high_14 - lowest_low_14) > 0,
                          ((highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)) * -100,
                          -50)  # neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_confirm_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) or opposite signal
            if williams_r[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) or opposite signal
            if williams_r[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R reversals with volume confirmation and range regime
            # Only trade in ranging markets (ADX < 25) where mean reversion works
            if volume_confirm_aligned[i] and adx_aligned[i] < 25:
                # Long: Williams %R crosses above -80 from below (oversold bounce)
                if williams_r[i] > -80 and williams_r[i-1] <= -80:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Williams %R crosses below -20 from above (overbought rejection)
                elif williams_r[i] < -20 and williams_r[i-1] >= -20:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals