# Your hypothesis: A weekly RSI divergence strategy on daily data, with volume confirmation and volatility filter
# Rationale: RSI divergence captures exhaustion in trends, works in both bull/bear markets by identifying reversals
# Weekly timeframe provides higher reliability, daily for execution
# Volume confirmation ensures institutional participation
# ATR-based volatility filter avoids choppy markets
# Target: ~15-25 trades/year by requiring multiple confluence factors

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Get daily data for context
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate weekly RSI (14-period)
    weekly_close = df_weekly['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_weekly = 100 - (100 / (1 + rs))
    rsi_weekly_aligned = align_htf_to_ltf(prices, df_weekly, rsi_weekly)
    
    # Calculate weekly RSI slope for divergence detection
    rsi_slope = np.gradient(rsi_weekly_aligned)
    
    # Calculate daily price slope for comparison
    daily_close = df_daily['close'].values
    price_slope = np.gradient(daily_close)
    price_slope_aligned = align_htf_to_ltf(prices, df_daily, price_slope)
    
    # Calculate daily ATR (14-period) for volatility filter
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    def wilder_smooth_tr(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_daily = wilder_smooth_tr(tr, 14)
    atr_daily_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Calculate daily average volume (20-period)
    volume_daily = df_daily['volume'].values
    vol_avg_20 = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        rsi_val = rsi_weekly_aligned[i]
        rsi_slope_val = rsi_slope[i]
        price_slope_val = price_slope_aligned[i]
        atr_val = atr_daily_aligned[i]
        vol_val = prices['volume'].iloc[i]
        vol_avg_val = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN or invalid
        if (np.isnan(rsi_val) or np.isnan(rsi_slope_val) or 
            np.isnan(price_slope_val) or np.isnan(atr_val) or 
            np.isnan(vol_avg_val) or atr_val <= 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Normalize slopes by ATR for comparison
        norm_rsi_slope = rsi_slope_val / (atr_val * 0.1)  # Scale factor
        norm_price_slope = price_slope_val / atr_val
        
        if position == 0:
            # Bullish divergence: RSI making higher low while price makes lower low
            # Bearish divergence: RSI making lower high while price makes higher high
            if (norm_rsi_slope > 0.1 and norm_price_slope < -0.1 and 
                rsi_val < 40 and vol_val > vol_avg_val * 1.5):
                # Bullish divergence + oversold + volume spike
                signals[i] = 0.25
                position = 1
            elif (norm_rsi_slope < -0.1 and norm_price_slope > 0.1 and 
                  rsi_val > 60 and vol_val > vol_avg_val * 1.5):
                # Bearish divergence + overbought + volume spike
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish divergence or RSI overbought
            if (norm_rsi_slope < -0.1 and norm_price_slope > 0.1) or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish divergence or RSI oversold
            if (norm_rsi_slope > 0.1 and norm_price_slope < -0.1) or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_WeeklyRSI_Divergence_VolumeFilter
# Uses weekly RSI divergence for high-probability reversals
# Requires volume confirmation (>1.5x average) to filter false signals
# Uses ATR-normalized slope comparison to avoid scale issues
# Session filter: 8-20 UTC to avoid low-volume periods
# Exits on opposite divergence or RSI extreme levels
# Designed for 1d timeframe with ~15-25 trades/year
name = "1d_WeeklyRSI_Divergence_VolumeFilter"
timeframe = "1d"
leverage = 1.0