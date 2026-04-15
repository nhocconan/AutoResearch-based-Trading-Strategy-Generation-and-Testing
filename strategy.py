#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d VWAP AND volume > 1.5x avg
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d VWAP AND volume > 1.5x avg
# Uses 1d ADX > 25 to filter for trending markets only, reducing whipsaws in ranging conditions
# Designed for moderate trade frequency (50-120/year) with strong directional bias in trends

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: VWAP and ADX for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # VWAP calculation
    vwap_1d = (np.cumsum(close_1d * volume_1d) / np.cumsum(volume_1d))
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # ADX calculation (14-period)
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === ELDER RAY CALCULATION (using 13-period EMA) ===
        # Calculate EMA13 for Bull/Bear Power
        ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
        bull_power = high - ema13
        bear_power = ema13 - low
        
        # Calculate rising/falling power (1-period change)
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_falling = bear_power[i] < bear_power[i-1]
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trending_regime = adx_1d_aligned[i] > 25
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (buying strength)
        # 2. Bull Power rising (momentum increasing)
        # 3. Price above 1d VWAP (intraday bullish bias)
        # 4. Volume confirmation
        # 5. Trending regime (ADX > 25)
        if (bull_power[i] > 0) and bull_power_rising and (close[i] > vwap_1d_aligned[i]) and vol_confirm and trending_regime:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power > 0 (selling strength) 
        # 2. Bear Power falling (momentum increasing to downside)
        # 3. Price below 1d VWAP (intraday bearish bias)
        # 4. Volume confirmation
        # 5. Trending regime (ADX > 25)
        elif (bear_power[i] > 0) and bear_power_falling and (close[i] < vwap_1d_aligned[i]) and vol_confirm and trending_regime:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_1dADX_VWAP_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0