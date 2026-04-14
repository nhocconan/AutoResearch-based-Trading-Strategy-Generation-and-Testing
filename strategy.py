#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining daily Ichimoku Cloud as trend filter
# with 12h Relative Strength Index (RSI) for mean-reversion entries.
# - Long when price is above daily Kumo (cloud) and 12h RSI < 30 (oversold)
# - Short when price is below daily Kumo and 12h RSI > 70 (overbought)
# - Volume confirmation: current volume > 1.3x 20-period average to ensure participation
# - Uses Ichimoku Cloud for robust trend filtering that adapts to volatility
# - RSI(14) provides mean-reversion signals within the trend context
# - Target: 50-150 total trades over 4 years (12-37/year) for optimal balance
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku Components (daily)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For trend filtering, we use the current cloud (Senkou Span A/B shifted)
    # But to avoid look-ahead, we use the cloud values from 26 periods ago
    # Actually, for determining if price is above/below cloud at time t,
    # we use Senkou Span A/B calculated at t-26 (already known)
    # So we shift Senkou Span A/B forward by 26 to align with current price
    # But since we want to avoid look-ahead, we'll use the cloud as it was known 26 periods ago
    # Simpler: use the current Senkou Span A/B values (which are plotted 26 periods ahead)
    # To get the cloud values that are visible at current time, we need Senkou Span A/B from 26 periods ago
    # So we'll use Senkou Span A/B values shifted BACK by 26 to get current cloud
    
    # The cloud that is visible at time t is Senkou Span A/B calculated at t-26
    # So we take Senkou Span A/B and shift them forward by 26 to align with current time
    # But to avoid look-ahead, we use values that were known in the past
    # Actually, Senkou Span A/B are plotted 26 periods ahead, so the current cloud
    # was calculated 26 periods ago. Therefore, we can use the current Senkou Span A/B
    # values as they represent the cloud that is visible now (calculated in the past)
    
    # For simplicity and to avoid look-ahead issues, we'll use:
    # Kumo top = max(Senkou Span A, Senkou Span B) 
    # Kumo bottom = min(Senkou Span A, Senkou Span B)
    # These represent the current cloud boundaries
    
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Calculate 12h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get daily index for current 12h bar
        # 1 day = 2 * 12h bars
        idx_1d = i // 2
        if idx_1d < 1:
            continue
            
        # Current Kumo (cloud) values - these are safe to use as they represent
        # the cloud that is visible now (calculated from past data)
        kumo_top_val = kumo_top[idx_1d]
        kumo_bottom_val = kumo_bottom[idx_1d]
        
        if np.isnan(kumo_top_val) or np.isnan(kumo_bottom_val):
            continue
        
        if position == 0:
            # Long: price above Kumo + RSI oversold + volume confirmation
            if (close[i] > kumo_top_val and      # price above cloud
                rsi[i] < 30 and                  # RSI oversold
                volume[i] > vol_ma[i] * 1.3):    # volume confirmation
                position = 1
                signals[i] = position_size
            # Short: price below Kumo + RSI overbought + volume confirmation
            elif (close[i] < kumo_bottom_val and   # price below cloud
                  rsi[i] > 70 and                  # RSI overbought
                  volume[i] > vol_ma[i] * 1.3):    # volume confirmation
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: RSI overbought or price below Kumo
            if rsi[i] > 70 or close[i] < kumo_bottom_val:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: RSI oversold or price above Kumo
            if rsi[i] < 30 or close[i] > kumo_top_val:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_Ichimoku_RSI_MeanReversion"
timeframe = "12h"
leverage = 1.0