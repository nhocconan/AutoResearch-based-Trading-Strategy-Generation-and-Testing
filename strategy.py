#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volume confirmation
# - Uses 4h HMA(21) for trend direction (long when price > HMA, short when price < HMA)
# - Uses 1d Camarilla levels (H3, L3) for breakout entries on 1h timeframe
# - Requires 1h volume > 1.5 * 20-period volume average for confirmation
# - Uses session filter (08-20 UTC) to avoid low-liquidity hours
# - Target: 15-30 trades/year on 1h timeframe (60-120 total over 4 years) to avoid fee drag
# - Works in bull markets via breakouts above resistance, in bear via breakdowns below support

name = "1h_4h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 21 or len(df_1d) < 5:
        return np.zeros(n)
    
    # 4h HMA(21) for trend filter
    close_4h = df_4h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_4h).rolling(window=half_len, min_periods=half_len).mean().values
    wma_full = pd.Series(close_4h).rolling(window=21, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_4h = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 1h ATR(14) for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: stoploss or mean reversion
            if close[i] < highest_high_since_entry - 2.0 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < l3_aligned[i]:  # Mean reversion exit (break below L3)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: stoploss or mean reversion
            if close[i] > lowest_low_since_entry + 2.0 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > h3_aligned[i]:  # Mean reversion exit (break above H3)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for breakout entries with trend and volume confirmation
            if close[i] > h3_aligned[i] and close[i] > hma_4h_aligned[i] and volume_confirm[i]:  # Break above H3 with uptrend
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.20
            elif close[i] < l3_aligned[i] and close[i] < hma_4h_aligned[i] and volume_confirm[i]:  # Break below L3 with downtrend
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.20
    
    return signals