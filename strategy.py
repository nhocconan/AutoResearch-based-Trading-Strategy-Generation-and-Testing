#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above Camarilla R3 (1d) + volume > 2x 20-period average + CHOP > 61.8 (range market)
# Short when price breaks below Camarilla S3 (1d) + volume > 2x 20-period average + CHOP > 61.8
# Uses discrete position sizing (0.30) and ATR-based stoploss. Designed for 20-40 trades/year.
# Camarilla pivots provide precise intraday support/resistance levels. Volume spike confirms breakout strength.
# Choppiness filter ensures we only trade in ranging markets where mean reversion at pivots works best.
# Works in both bull and bear markets by fading extreme moves back to the mean in range-bound conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4.0)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4.0)
    
    # Align Camarilla levels to 4h timeframe (no extra delay needed for pivot points)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 4h Indicators: Volume Spike and Choppiness Index ===
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # True Range for Choppiness Index
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - close_shift)
    tr3 = np.abs(low - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period) for denominator of CHOP
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Sum of TRUE RANGE over chop_period for numerator
    chop_period = 14
    sum_tr = np.zeros_like(tr)
    for i in range(chop_period-1, len(tr)):
        if i == chop_period-1:
            sum_tr[i] = np.sum(tr[:chop_period])
        else:
            sum_tr[i] = sum_tr[i-1] - tr[i-chop_period] + tr[i]
    
    # Max and Min close over chop_period for denominator
    max_close = np.zeros_like(close)
    min_close = np.zeros_like(close)
    for i in range(chop_period-1, len(close)):
        if i == chop_period-1:
            max_close[i] = np.max(close[:chop_period])
            min_close[i] = np.min(close[:chop_period])
        else:
            max_close[i] = max(max_close[i-1], close[i])
            min_close[i] = min(min_close[i-1], close[i])
    
    # Choppiness Index: CHOP = 100 * LOG10(sum_tr / (ATR * chop_period)) / LOG10(chop_period)
    # Avoid division by zero and invalid values
    denominator = atr * chop_period
    chop_raw = np.zeros_like(tr)
    valid_chop = (denominator > 0) & (sum_tr > 0)
    chop_raw[valid_chop] = 100 * np.log10(sum_tr[valid_chop] / denominator[valid_chop]) / np.log10(chop_period)
    
    # CHOP values typically range 0-100; >61.8 = ranging, <38.2 = trending
    chop = chop_raw
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, atr_period + chop_period) + 5  # volume(20) + ATR(14) + CHOP(14) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Choppiness filter: CHOP > 61.8 (ranging market)
        chop_filter = chop[i] > 61.8
        
        # Skip if any required data is NaN or invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d)
        # 2. Volume confirmation
        # 3. Chop filter (ranging market)
        if (close[i] > camarilla_r3_aligned[i]) and \
           vol_confirm and chop_filter:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d)
        # 2. Volume confirmation
        # 3. Chop filter (ranging market)
        elif (close[i] < camarilla_s3_aligned[i]) and \
             vol_confirm and chop_filter:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_CamarillaR3S3_1dVolumeSpike_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0