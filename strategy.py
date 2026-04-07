#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversals with 1-day trend filter and volume confirmation
# Fade at R3/S3 levels (mean reversion), breakout continuation at R4/S4 (trend following)
# Use 1-day trend (close vs EMA50) to filter direction: only fade against trend, only breakout with trend
# Volume > 1.5x average confirms participation
# Target: 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear by adapting to regime: fades in range, breaks in trend

name = "6h_camarilla_pivot_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.1 / 2)
    # R3 = C + ((H - L) * 1.1 / 4)
    # S3 = C - ((H - L) * 1.1 / 4)
    # S4 = C - ((H - L) * 1.1 / 2)
    pp = (high_1d + low_1d + close_1d) / 3
    r4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    s4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align pivot levels to 6h timeframe (use previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1-day EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6-day volume average for confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    volume_6h = df_6h['volume'].values
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma_6h_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: reverse at S3 (mean reversion) or break above R4 (trend)
            elif close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: reverse at R3 (mean reversion) or break below S4 (trend)
            elif close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Fade at R3/S3: sell at R3 in uptrend, buy at S3 in downtrend
            # Breakout at R4/S4: buy at R4 in uptrend, sell at S4 in downtrend
            if volume[i] > 1.5 * volume_ma_6h_aligned[i]:
                # Fade logic: sell at R3 when above EMA (uptrend), buy at S3 when below EMA (downtrend)
                if close[i] > r3_aligned[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                elif close[i] < s3_aligned[i] and close[i] < ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakout logic: buy at R4 when above EMA, sell at S4 when below EMA
                elif close[i] > r4_aligned[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < s4_aligned[i] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals