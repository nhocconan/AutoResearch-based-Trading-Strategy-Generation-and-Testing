#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) for trend direction,
# combined with 4h volume spike and ADX regime filter. Alligator provides smoothed trend
# via SMAs, volume confirms momentum strength, ADX > 25 ensures trending market to avoid
# whipsaws. Long when Lips > Teeth > Jaw (bullish alignment), price > VWAP, volume > 1.5x
# 20-period average, ADX > 25. Short when Lips < Teeth < Jaw (bearish alignment), price < VWAP,
# volume > 1.5x 20-period average, ADX > 25. Exit when Alligator alignment breaks or ADX < 20.
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Alligator is effective in both bull and bear markets as it identifies trend strength and direction.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d Indicators: Williams Alligator ===
    # Jaw (Blue): 13-period SMMA, shifted 8 bars
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth (Red): 8-period SMMA, shifted 5 bars
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips (Green): 5-period SMMA, shifted 3 bars
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Alligator alignment: 1 = bullish (Lips > Teeth > Jaw), -1 = bearish (Lips < Teeth < Jaw), 0 = entwined
    alligator_align = np.zeros_like(close_1d)
    bullish = (lips > teeth) & (teeth > jaw)
    bearish = (lips < teeth) & (teeth < jaw)
    alligator_align[bullish] = 1
    alligator_align[bearish] = -1
    
    # Align 1d Alligator alignment to 4h timeframe
    alligator_align_aligned = align_htf_to_ltf(prices, df_1d, alligator_align)
    
    # Get 4h data once before loop for VWAP, volume, and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Typical price for VWAP
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    vp = typical_price_4h * volume_4h
    
    # Cumulative VWAP (reset daily)
    cum_vp = np.cumsum(vp)
    cum_vol = np.cumsum(volume_4h)
    vwap = np.divide(cum_vp, cum_vol, out=np.zeros_like(cum_vp), where=cum_vol!=0)
    
    # Volume moving average (20-period) on 4h
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[:period]) / period
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = wilders_smoothing(dx, 14)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(alligator_align_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_ma_20_4h[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        alligator_val = alligator_align_aligned[i]
        vwap_val = vwap[i]
        vol_ma_val = vol_ma_20_4h[i]
        adx_val = adx[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Alligator alignment turns bearish/entwined or ADX drops below 20
            if alligator_val <= 0 or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Alligator alignment turns bullish/entwined or ADX drops below 20
            if alligator_val >= 0 or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: Alligator alignment must be non-zero (trending)
            trend_filter = alligator_val != 0
            
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma_val
            
            # Regime filter: ADX > 25 (strong trend)
            regime_filter = adx_val > 25
            
            # Price filter: price must be on correct side of VWAP
            price_filter_long = price > vwap_val
            price_filter_short = price < vwap_val
            
            # LONG: Alligator bullish, price > VWAP, volume spike, strong trend
            if (alligator_val > 0) and price_filter_long and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Alligator bearish, price < VWAP, volume spike, strong trend
            elif (alligator_val < 0) and price_filter_short and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dAlligator_VWAP_VolumeSpike_ADXFilter_V1"
timeframe = "4h"
leverage = 1.0