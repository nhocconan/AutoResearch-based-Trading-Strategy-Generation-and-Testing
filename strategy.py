#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Daily data for weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close from daily data (simplified: using last 5 days)
    # For true weekly data, we would need weekly aggregation, but we approximate
    # using the most recent 5 trading days as proxy for weekly range
    if len(high_1d) < 5:
        # Not enough data for weekly approximation
        weekly_high = np.full_like(high_1d, np.nan)
        weekly_low = np.full_like(high_1d, np.nan)
        weekly_close = np.full_like(close_1d, np.nan)
    else:
        # Use rolling window of 5 days to approximate weekly range
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point calculation (using weekly high/low/close)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = weekly_pivot + (weekly_high - weekly_low) * 1.1 / 12
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low) * 1.1 / 6
    weekly_r3 = weekly_pivot + (weekly_high - weekly_low) * 1.1 / 4
    weekly_s1 = weekly_pivot - (weekly_high - weekly_low) * 1.1 / 12
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low) * 1.1 / 6
    weekly_s3 = weekly_pivot - (weekly_high - weekly_low) * 1.1 / 4
    
    # Weekly EMA34 for trend filter
    weekly_ema_34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe (primary)
    wp_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    wr1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    wr2_6h = align_htf_to_ltf(prices, df_1d, weekly_r2)
    wr3_6h = align_htf_to_ltf(prices, df_1d, weekly_r3)
    ws1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    ws2_6h = align_htf_to_ltf(prices, df_1d, weekly_s2)
    ws3_6h = align_htf_to_ltf(prices, df_1d, weekly_s3)
    wema_34_6h = align_htf_to_ltf(prices, df_1d, weekly_ema_34)
    
    # 6h ATR(20) for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.8 * vol_ma20  # Volume surge threshold
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(wp_6h[i]) or np.isnan(wr1_6h[i]) or np.isnan(wr2_6h[i]) or np.isnan(wr3_6h[i]) or
            np.isnan(ws1_6h[i]) or np.isnan(ws2_6h[i]) or np.isnan(ws3_6h[i]) or
            np.isnan(wema_34_6h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above WS3 with volume surge, above weekly EMA34
            if (close[i] > ws3_6h[i] and vol_surge[i] and close[i] > wema_34_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below WR3 with volume surge, below weekly EMA34
            elif (close[i] < wr3_6h[i] and vol_surge[i] and close[i] < wema_34_6h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses weekly pivot or volatility drops significantly
            if position == 1:
                if close[i] < wp_6h[i] or atr[i] < 0.4 * atr[i-1]:  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > wp_6h[i] or atr[i] < 0.4 * atr[i-1]:  # Volatility drop filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_WS3_WR3_Breakout_1dEMA34_Trend_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0