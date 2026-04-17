#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Camarilla H3/L3 breakout with volume confirmation and ATR-based volatility filter.
- Long when price breaks above 1d Camarilla H3 level + volume > 1.5x 24-period 12h volume MA + ATR(12h) > 0.5 * ATR(24-period MA)
- Short when price breaks below 1d Camarilla L3 level + volume > 1.5x 24-period 12h volume MA + ATR(12h) > 0.5 * ATR(24-period MA)
- Exit when price closes below/above 1d EMA34 (trend change) or opposite Camarilla level
- Fixed position size 0.25 to manage drawdown in bear markets
- Uses proven edge: Camarilla levels (intraday support/resistance) + volume spike + volatility filter
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
- Works in bull markets (buying breakouts with uptrend) and bear markets (selling breakdowns with downtrend)
- ATR filter ensures we only trade during sufficient volatility, avoiding choppy periods
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate typical price for 1d
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels: H3/L3 = typical_price ± 1.1 * (high - low) / 2
    camarilla_h3_1d = typical_price_1d + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_l3_1d = typical_price_1d - 1.1 * (high_1d - low_1d) / 2.0
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (24-period = 12d equivalent) on 12h for confirmation
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR calculation on 12h for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=24, min_periods=24).mean().values
    
    # Align all HTF indicators to primary timeframe (12h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    volume_ma_24_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_24)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_24_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(atr_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_ma = volume_ma_24_aligned[i]
        atr_val = atr_aligned[i]
        atr_ma_val = atr_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Volatility filter: only trade when ATR > 50% of its MA (avoid choppy periods)
        if atr_val <= 0.5 * atr_ma_val:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakouts with volume confirmation
            # Long: price breaks above Camarilla H3 + volume spike
            if price > camarilla_h3 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla L3 + volume spike
            elif price < camarilla_l3 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 1d EMA34 (trend change) or below Camarilla L3
            if price < ema_34_val or price < camarilla_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 1d EMA34 (trend change) or above Camarilla H3
            if price > ema_34_val or price > camarilla_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_VolumeSpike_ATRFilter_1dEMA34"
timeframe = "12h"
leverage = 1.0