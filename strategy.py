#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d trend filter and volume confirmation.
# Uses 4h EMA50 and 1d EMA100 for trend direction, 1h for Camarilla pivot breakout timing.
# Includes 1d RSI filter to avoid extremes and session filter (08-20 UTC) to reduce noise.
# Designed for low trade frequency (15-35/year) to minimize fee drag in choppy markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Trend Filter ===
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d Indicators: Higher Timeframe Bias ===
    # 1d EMA(100) for long-term trend
    ema_100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # 1d RSI(14) for overbought/oversold filter (avoid extremes)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_100_1d_aligned[i]) or
            np.isnan(rsi_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === 1h Camarilla Pivot Calculation (using prior 1h bar) ===
        # Need prior bar's OHLC for today's pivot levels
        if i == 0:
            signals[i] = 0.0
            continue
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        
        pivot = (phigh + plow + pclose) / 3.0
        range_ = phigh - plow
        
        # Camarilla levels
        r3 = pivot + (range_ * 1.1 / 4.0)
        s3 = pivot - (range_ * 1.1 / 4.0)
        
        # Volume filter: current 1h volume > 1.5x 20-period 1h volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. 4h price above EMA50 (bullish 4h trend)
        # 2. 1d price above EMA100 (bullish long-term trend)
        # 3. 1d RSI between 30 and 70 (not extreme)
        # 4. Price breaks above Camarilla R3 (breakout)
        # 5. Volume confirmation
        if (close[i] > ema_50_4h_aligned[i] and
            close[i] > ema_100_1d_aligned[i] and
            30 < rsi_14_1d_aligned[i] < 70 and
            close[i] > r3 and
            vol_confirm):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. 4h price below EMA50 (bearish 4h trend)
        # 2. 1d price below EMA100 (bearish long-term trend)
        # 3. 1d RSI between 30 and 70 (not extreme)
        # 4. Price breaks below Camarilla S3 (breakdown)
        # 5. Volume confirmation
        elif (close[i] < ema_50_4h_aligned[i] and
              close[i] < ema_100_1d_aligned[i] and
              30 < rsi_14_1d_aligned[i] < 70 and
              close[i] < s3 and
              vol_confirm):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R3S3_EMA50_EMA100_RSI_VolFilter_v1"
timeframe = "1h"
leverage = 1.0