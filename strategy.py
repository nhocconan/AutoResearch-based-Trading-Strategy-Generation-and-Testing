#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot with 4h volume confirmation and 1d trend filter
# - Enter long when 1h price touches Camarilla L3 support AND 4h volume > 1.5x 20-period volume SMA AND 1d close > 1d EMA50
# - Enter short when 1h price touches Camarilla H3 resistance AND 4h volume > 1.5x 20-period volume SMA AND 1d close < 1d EMA50
# - Exit: price moves to opposite Camarilla level (L3 for shorts, H3 for longs) or opposite pivot touch
# - Session filter: only trade 08-20 UTC to avoid low-volume Asian session
# - Position size: 0.20 (20% of capital) to manage drawdown in bear markets
# - Target: 15-35 trades/year to minimize fee drag while capturing high-probability reversals

name = "1h_4h_1d_camarilla_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop for volume confirmation (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return signals
    
    # Load 1d data ONCE before loop for trend filter and Camarilla calculation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute Camarilla levels for 1d data (based on previous day's OHLC)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_range = high_1d - low_1d
    h3 = close_1d + 1.125 * camarilla_range  # H3 resistance
    l3 = close_1d - 1.125 * camarilla_range  # L3 support
    
    # Align Camarilla levels to 1h timeframe (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Pre-compute volume SMA for 4h data (20-period)
    volume_4h = df_4h['volume'].values
    volume_sma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    # Pre-compute EMA50 for 1d close (trend filter)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 1d close aligned for trend comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    for i in range(50, n):  # Start after 50-bar warmup for EMA50
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_sma_20_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.5x 20-period volume SMA
        volume_4h_current = df_4h['volume'].values
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h_current)
        vol_confirm = volume_4h_aligned[i] > 1.5 * volume_sma_20_4h_aligned[i]
        
        # Trend filter: 1d close vs EMA50
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Camarilla pivot touch signals (using 1h high/low for touch detection)
        touch_h3 = high[i] >= h3_aligned[i] and low[i] <= h3_aligned[i]  # Price touched H3 level
        touch_l3 = high[i] >= l3_aligned[i] and low[i] <= l3_aligned[i]  # Price touched L3 level
        
        # Trading logic
        if vol_confirm:
            # Long: L3 touch in uptrend
            if touch_l3 and uptrend:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.20
                else:
                    signals[i] = 0.20
            # Short: H3 touch in downtrend
            elif touch_h3 and downtrend:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.20
                else:
                    signals[i] = -0.20
            else:
                # Check for exits: exit when price touches opposite level
                if position == 1 and touch_h3:  # Exit long when price reaches H3
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and touch_l3:  # Exit short when price reaches L3
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
        else:
            # No volume confirmation: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals