#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w RSI divergence with 1d volume confirmation
# - Bearish divergence: price makes higher high while 1w RSI makes lower high → short
# - Bullish divergence: price makes lower low while 1w RSI makes higher low → long
# - Uses 1d volume spike to confirm reversal strength
# - Filters trades to only occur when price is outside 1d Bollinger Bands (2,2) for extremity
# - Designed to work in ranging markets (divergences at extremes) and avoid chop
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_RSIDivergence_1dVolume_BollingerFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for RSI calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1w close
    rsi_period = 14
    delta = np.diff(df_1w['close'], prepend=df_1w['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1w RSI to 12h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Get 1d data for volume and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(df_1d['close']).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(df_1d['close']).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # Align 1d indicators to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # 1d volume and its 20-period moving average
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Volume spike: current volume > 1.5 * 20-period average
    volume_spike = df_1d['volume'].values > (1.5 * vol_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low
            # Look back 3 periods for swing points
            if i >= 3:
                # Price lower low
                price_lower_low = (low[i] < low[i-1]) and (low[i-1] < low[i-2])
                # RSI higher low
                rsi_higher_low = (rsi_1w_aligned[i] > rsi_1w_aligned[i-1]) and (rsi_1w_aligned[i-1] > rsi_1w_aligned[i-2])
                
                if price_lower_low and rsi_higher_low:
                    # Price must be below lower Bollinger Band for extremity
                    if close[i] < lower_band_aligned[i]:
                        # Volume spike confirmation
                        if volume_spike_aligned[i]:
                            signals[i] = 0.25
                            position = 1
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if i >= 3:
                # Price higher high
                price_higher_high = (high[i] > high[i-1]) and (high[i-1] > high[i-2])
                # RSI lower high
                rsi_lower_high = (rsi_1w_aligned[i] < rsi_1w_aligned[i-1]) and (rsi_1w_aligned[i-1] < rsi_1w_aligned[i-2])
                
                if price_higher_high and rsi_lower_high:
                    # Price must be above upper Bollinger Band for extremity
                    if close[i] > upper_band_aligned[i]:
                        # Volume spike confirmation
                        if volume_spike_aligned[i]:
                            signals[i] = -0.25
                            position = -1
        
        elif position == 1:
            # Exit long: price returns to middle of Bollinger Bands (SMA) or RSI overbought
            if close[i] >= sma_aligned[i] if 'sma_aligned' in locals() else False:  # Will define sma_aligned below
                signals[i] = 0.0
                position = 0
            elif rsi_1w_aligned[i] > 70:  # RSI overbought
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle of Bollinger Bands or RSI oversold
            if close[i] <= sma_aligned[i] if 'sma_aligned' in locals() else False:
                signals[i] = 0.0
                position = 0
            elif rsi_1w_aligned[i] < 30:  # RSI oversold
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    # Recalculate sma_aligned for exit conditions (moved inside loop for clarity)
    sma_aligned = align_htf_to_ltf(prices, df_1d, sma)
    
    # Re-run loop with proper sma_aligned reference
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(sma_aligned[i]) or np.isnan(vol_ma_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            if i >= 3:
                price_lower_low = (low[i] < low[i-1]) and (low[i-1] < low[i-2])
                rsi_higher_low = (rsi_1w_aligned[i] > rsi_1w_aligned[i-1]) and (rsi_1w_aligned[i-1] > rsi_1w_aligned[i-2])
                if price_lower_low and rsi_higher_low:
                    if close[i] < lower_band_aligned[i]:
                        if volume_spike_aligned[i]:
                            signals[i] = 0.25
                            position = 1
            
            if i >= 3:
                price_higher_high = (high[i] > high[i-1]) and (high[i-1] > high[i-2])
                rsi_lower_high = (rsi_1w_aligned[i] < rsi_1w_aligned[i-1]) and (rsi_1w_aligned[i-1] < rsi_1w_aligned[i-2])
                if price_higher_high and rsi_lower_high:
                    if close[i] > upper_band_aligned[i]:
                        if volume_spike_aligned[i]:
                            signals[i] = -0.25
                            position = -1
        
        elif position == 1:
            if close[i] >= sma_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif rsi_1w_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if close[i] <= sma_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif rsi_1w_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals