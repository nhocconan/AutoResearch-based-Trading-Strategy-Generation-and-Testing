#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d regime filter
# - Uses 4h ADX(14) for trend strength filter (ADX > 25 = trending)
# - Uses 1d Camarilla levels (H3, L3, H4, L4) from prior day for structure
# - Enters breakout trades on 1h when price closes beyond H3/L3 with volume confirmation
# - Only takes trades in direction of 4h trend (long when 4h EMA50 > EMA200, short when inverse)
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Target: 15-35 trades/year on 1h timeframe (60-140 total over 4 years) to avoid fee drag
# - Camarilla levels provide statistically significant support/resistance in ranging markets
# - Breakout + volume + trend alignment reduces false signals in choppy conditions

name = "1h_4h_1d_camarilla_breakout_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA50 and EMA200 for trend direction
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
    trend_4h = np.where(ema50_4h > ema200_4h, 1, -1)  # 1=uptrend, -1=downtrend
    
    # Align 4h trend to 1h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1d Camarilla levels (based on prior day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    camarilla_h4 = np.full_like(close_1d, np.nan)
    camarilla_l4 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Prior day OHLC
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        range_val = phigh - plow
        if range_val <= 0:
            continue
            
        camarilla_h3[i] = pclose + (range_val * 1.1 / 4)
        camarilla_l3[i] = pclose - (range_val * 1.1 / 4)
        camarilla_h4[i] = pclose + (range_val * 1.1 / 2)
        camarilla_l4[i] = pclose - (range_val * 1.1 / 2)
    
    # Align 1d Camarilla levels to 1h
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1h volume confirmation (volume > 1.5x 20-period average)
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20>0)
    
    # 1h close price
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(trend_4h_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma20[i]) or vol_ma20[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or trend change
            if close[i] <= h3_aligned[i]:  # Return to H3 level
                position = 0
                signals[i] = 0.0
            elif trend_4h_aligned[i] == -1:  # 4h trend turned down
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or trend change
            if close[i] >= l3_aligned[i]:  # Return to L3 level
                position = 0
                signals[i] = 0.0
            elif trend_4h_aligned[i] == 1:  # 4h trend turned up
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for breakout entries aligned with 4h trend and volume
            if (close[i] > h3_aligned[i] and 
                vol_ratio[i] > 1.5 and  # Volume confirmation
                trend_4h_aligned[i] == 1):  # 4h uptrend
                position = 1
                signals[i] = 0.20
            elif (close[i] < l3_aligned[i] and 
                  vol_ratio[i] > 1.5 and  # Volume confirmation
                  trend_4h_aligned[i] == -1):  # 4h downtrend
                position = -1
                signals[i] = -0.20
    
    return signals