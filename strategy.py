#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h volume spike filter and 1d ADX trend filter
# Long when price breaks above 1h Camarilla R1 + 4h volume > 2x 20-period avg + 1d ADX > 20
# Short when price breaks below 1h Camarilla S1 + 4h volume > 2x 20-period avg + 1d ADX > 20
# Uses session filter (08-20 UTC) and discrete position sizing (0.20) to minimize fee churn.
# Designed for low trade frequency (15-35/year) to avoid fee drag. Camarilla levels provide
# intraday support/resistance; volume confirms breakout strength; ADX ensures trending market.
# Works in bull markets (breakout continuation) and bear markets (strong intraday moves).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop (for volume filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop (for ADX trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicator: Volume SMA (20-period) ===
    vol_4h = df_4h['volume'].values
    vol_sma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_sma_20_4h)
    
    # === 1d Indicator: ADX (trend strength filter) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components: +DM, -DM, TR
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    high_1d_shift[0] = high_1d[0]
    low_1d_shift[0] = low_1d[0]
    
    plus_dm = np.where((high_1d - high_1d_shift) > (low_1d_shift - low_1d), 
                       np.maximum(high_1d - high_1d_shift, 0), 0)
    minus_dm = np.where((low_1d_shift - low_1d) > (high_1d - high_1d_shift), 
                        np.maximum(low_1d_shift - low_1d, 0), 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(plus_dm)
    minus_di = np.zeros_like(minus_dm)
    
    # Smooth +DM and -DM
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    for i in range(period, len(plus_dm)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # Wilder's smoothing for ADX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period-1:2*period])
    for i in range(2*period, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 2*period) + 5  # Camarilla needs prior day, ADX(28) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Calculate 1h Camarilla levels from previous day's OHLC
        # Camarilla levels use daily OHLC, so we need to get previous day's values
        # For 1h timeframe, we look back 24 hours (24 bars) for previous day
        if i < 24:
            signals[i] = 0.0
            continue
            
        prev_day_high = high[i-24:i].max()   # High of previous 24h (1 day)
        prev_day_low = low[i-24:i].min()     # Low of previous 24h (1 day)
        prev_day_close = close[i-24:i].mean() # Close of previous day (approximate)
        
        # Camarilla levels
        camarilla_range = prev_day_high - prev_day_low
        r1 = prev_day_close + camarilla_range * 1.1 / 12
        s1 = prev_day_close - camarilla_range * 1.1 / 12
        
        # Volume filter: current 4h volume > 2x 20-period 4h volume SMA
        vol_confirm = vol_sma_20_4h_aligned[i] > 0 and volume[i] > (vol_sma_20_4h_aligned[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(r1) or np.isnan(s1) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_sma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1h Camarilla R1
        # 2. Volume confirmation (4h volume spike)
        # 3. Trend (1d ADX > 20)
        if (close[i] > r1) and vol_confirm and (adx_aligned[i] > 20):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1h Camarilla S1
        # 2. Volume confirmation (4h volume spike)
        # 3. Trend (1d ADX > 20)
        elif (close[i] < s1) and vol_confirm and (adx_aligned[i] > 20):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R1S1_4hVolumeSpike_1dADX20_v1"
timeframe = "1h"
leverage = 1.0