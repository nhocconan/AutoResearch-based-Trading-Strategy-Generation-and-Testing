#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 12h ADX trend filter + volume confirmation
# - Long when price breaks above Camarilla H3 level AND 12h ADX > 25 (trending market) AND volume > 1.5x 20-period average
# - Short when price breaks below Camarilla L3 level AND 12h ADX > 25 AND volume > 1.5x 20-period average
# - Exit when price returns to Camarilla PIVOT level (mean reversion to equilibrium)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Camarilla levels provide institutional support/resistance that work in both trending and ranging markets
# - ADX filter ensures we trade only when there is a strong trend, reducing false breakouts in ranging markets
# - Volume confirmation reduces false breakouts

name = "4h_12h_camarilla_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 12h ADX(14) for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_12h - np.roll(high_12h, 1)
    minus_dm = np.roll(low_12h, 1) - low_12h
    plus_dm[0] = 0
    minus_dm[0] = 0
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smoothed TR, PlusDM, MinusDM (Wilder's smoothing)
    tr_14 = np.zeros_like(tr)
    plus_dm_14 = np.zeros_like(plus_dm)
    minus_dm_14 = np.zeros_like(minus_dm)
    
    tr_14[13] = np.mean(tr[1:14])
    plus_dm_14[13] = np.mean(plus_dm[1:14])
    minus_dm_14[13] = np.mean(minus_dm[1:14])
    
    for i in range(14, len(tr)):
        tr_14[i] = (tr_14[i-1] * 13 + tr[i]) / 14
        plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
    
    # DI+ and DI-
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    dx = np.where(np.isnan(dx), 0, dx)
    
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX value (after 2*14 periods)
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # ADX trend filter: trending when ADX > 25
    adx_trend = adx > 25
    
    # Align HTF indicators to 4h timeframe
    adx_trend_aligned = align_htf_to_ltf(prices, df_12h, adx_trend)
    
    # Pre-compute 4h Camarilla levels from previous period's OHLC
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
        # Skip if any required data is invalid
        if (np.isnan(pivot[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_trend_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND trending market AND volume spike
            if (close[i] > camarilla_h3[i] and 
                adx_trend_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND trending market AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  adx_trend_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
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
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals