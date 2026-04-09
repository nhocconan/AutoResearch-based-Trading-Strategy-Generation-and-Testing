#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation and ATR regime filter
# Camarilla R3/S3 act as mean reversion zones in ranging markets, R4/S4 as breakout levels in trending markets
# Volume confirmation ensures institutional participation
# ATR regime filter (current ATR > 20-period mean) avoids low volatility chop
# Works in both bull and bear markets by adapting to regime (mean reversion in range, breakout in trend)
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1d_camarilla_atr_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR (14-period) for regime filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Camarilla pivot levels
    pivot = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3.0
    range_ = high_1d[-1] - low_1d[-1]
    r4 = close_1d[-1] + range_ * 1.1 / 2
    r3 = close_1d[-1] + range_ * 1.1 / 4
    s3 = close_1d[-1] - range_ * 1.1 / 4
    s4 = close_1d[-1] - range_ * 1.1 / 2
    
    # For historical Camarilla levels, we need to calculate for each day
    # Use rolling window to get daily high/low/close for each point in time
    df_1d_df = pd.DataFrame({'high': high_1d, 'low': low_1d, 'close': close_1d}, index=df_1d.index)
    rolling_high = df_1d_df['high'].rolling(window=1, min_periods=1).max()
    rolling_low = df_1d_df['low'].rolling(window=1, min_periods=1).min()
    rolling_close = df_1d_df['close'].rolling(window=1, min_periods=1).last()
    
    # Calculate Camarilla for each day using previous day's OHLC
    prev_high = rolling_high.shift(1)
    prev_low = rolling_low.shift(1)
    prev_close = rolling_close.shift(1)
    
    pivot_daily = (prev_high + prev_low + prev_close) / 3.0
    range_daily = prev_high - prev_low
    r4_daily = prev_close + range_daily * 1.1 / 2
    r3_daily = prev_close + range_daily * 1.1 / 4
    s3_daily = prev_close - range_daily * 1.1 / 4
    s4_daily = prev_close - range_daily * 1.1 / 2
    
    # Align Camarilla levels and ATR regime to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_daily.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_daily.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_daily.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_daily.values)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, (atr_14 > atr_ma_20).astype(float).values)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr_regime_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x average 6h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # ATR regime filter: only trade when current ATR > 20-period average (avoid low-vol chop)
        atr_regime = bool(atr_regime_aligned[i])
        
        if not volume_confirmed or not atr_regime:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to S3 or stop at S4 breakdown
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_aligned[i]:  # Stop loss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to R3 or stop at R4 breakout
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_aligned[i]:  # Stop loss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Mean reversion at R3/S3 with volume confirmation
            # Breakout continuation at R4/S4 with volume confirmation
            if volume_confirmed:
                if close[i] <= r3_aligned[i] and close[i] >= s3_aligned[i]:
                    # In range between R3 and S3 - look for mean reversion signals
                    # Long near S3, short near R3
                    if close[i] <= s3_aligned[i] * 1.002:  # Near S3 (0.2% buffer)
                        position = 1
                        signals[i] = position_size
                    elif close[i] >= r3_aligned[i] * 0.998:  # Near R3 (0.2% buffer)
                        position = -1
                        signals[i] = -position_size
                elif close[i] > r4_aligned[i]:
                    # Breakout above R4 - go long
                    position = 1
                    signals[i] = position_size
                elif close[i] < s4_aligned[i]:
                    # Breakdown below S4 - go short
                    position = -1
                    signals[i] = -position_size
    
    return signals