#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h ATR regime filter and volume confirmation
# - Long when price breaks above Camarilla H3 level on 1h AND 4h ATR(14) < 20-period median ATR (low volatility regime) AND volume > 2.0x 20-period average
# - Short when price breaks below Camarilla L3 level on 1h AND 4h ATR(14) < 20-period median ATR AND volume > 2.0x 20-period average
# - Exit when price returns to Camarilla PIVOT level (mean reversion to equilibrium)
# - Uses discrete position sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Uses 4h for signal direction (regime filter) and 1h only for entry timing
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Camarilla levels provide institutional support/resistance that work in both trending and ranging markets
# - ATR filter ensures we trade during low volatility periods when breakouts are more reliable
# - Volume confirmation reduces false breakouts
# - Discrete sizing (0.20) minimizes fee churn from frequent small changes

name = "1h_4h_camarilla_atr_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute 4h ATR(14) for regime filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range calculation
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_4h = np.zeros_like(tr)
    atr_4h[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_4h[i] = (atr_4h[i-1] * 13 + tr[i]) / 14
    
    # ATR regime: low volatility when current ATR < median of last 20 ATR values
    atr_ma_20 = pd.Series(atr_4h).rolling(window=20, min_periods=20).mean().values
    atr_median_20 = pd.Series(atr_4h).rolling(window=20, min_periods=20).median().values
    low_vol_regime = atr_4h < atr_median_20
    
    # Align HTF indicators to 1h timeframe
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_4h, low_vol_regime)
    
    # Pre-compute 1h Camarilla levels from previous period's OHLC
    # Camarilla levels use previous period's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h3 = pivot + (range_hl * 1.1 / 4)
    camarilla_l3 = pivot - (range_hl * 1.1 / 4)
    camarilla_h4 = pivot + (range_hl * 1.1 / 2)
    camarilla_l4 = pivot - (range_hl * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(pivot[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(low_vol_regime_aligned[i]) or
            not in_session[i]):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND low volatility regime AND volume spike
            if (close[i] > camarilla_h3[i] and 
                low_vol_regime_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below Camarilla L3 AND low volatility regime AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  low_vol_regime_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price returns to pivot level (mean reversion to equilibrium)
            exit_long = (position == 1 and close[i] <= pivot[i])
            exit_short = (position == -1 and close[i] >= pivot[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals