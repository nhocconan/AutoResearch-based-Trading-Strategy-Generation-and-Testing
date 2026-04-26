#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
In trending markets (price > 12h EMA50), buy breakouts above R1 and sell breakdowns below S1.
In ranging markets, avoid false breakouts using volume spike filter (volume > 1.5x 20-period average).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades per year.
Works in bull/bear via trend filter: only follow breakouts in direction of 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ATR for stoploss (using 14-period ATR)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Calculate 12h EMA50 for HTF trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    htf_trend = np.where(close > ema_50_12h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate volume spike filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(htf_trend[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Camarilla levels for TODAY (using previous day's OHLC)
        # For 4h data, we need to group by day - use open_time to get date
        current_date = prices.iloc[i]['open_time'].date()
        
        # Get previous day's data (all 4h bars from yesterday)
        if i >= 6:  # Need at least 6 bars back to ensure we have yesterday's data
            # Find start of yesterday's session
            lookback = i
            while lookback > 0 and prices.iloc[lookback]['open_time'].date() == current_date:
                lookback -= 1
            
            if lookback >= 0 and lookback < i:
                # Yesterday's data is from lookback+1 to i-1 (inclusive)
                yesterday_high = high[lookback+1:i].max()
                yesterday_low = low[lookback+1:i].min()
                yesterday_close = close[i-1]  # Previous bar's close
                
                # Calculate Camarilla levels
                range_val = yesterday_high - yesterday_low
                if range_val > 0:
                    R1 = yesterday_close + (range_val * 1.1 / 12)
                    S1 = yesterday_close - (range_val * 1.1 / 12)
                    
                    # Breakout conditions with volume confirmation and trend filter
                    if htf_trend[i] == 1:  # Uptrend - only look for longs
                        if close[i] > R1 and volume_spike[i]:
                            if position != 1:
                                signals[i] = 0.25
                                position = 1
                            else:
                                signals[i] = 0.25
                        elif position == 1 and close[i] < ema_50_12h_aligned[i]:
                            # Exit if price crosses below 12h EMA50
                            signals[i] = 0.0
                            position = 0
                        else:
                            # Hold current position
                            if position == 0:
                                signals[i] = 0.0
                            elif position == 1:
                                signals[i] = 0.25
                    else:  # Downtrend - only look for shorts
                        if close[i] < S1 and volume_spike[i]:
                            if position != -1:
                                signals[i] = -0.25
                                position = -1
                            else:
                                signals[i] = -0.25
                        elif position == -1 and close[i] > ema_50_12h_aligned[i]:
                            # Exit if price crosses above 12h EMA50
                            signals[i] = 0.0
                            position = 0
                        else:
                            # Hold current position
                            if position == 0:
                                signals[i] = 0.0
                            elif position == -1:
                                signals[i] = -0.25
                else:
                    # Invalid range calculation - hold position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            else:
                # Not enough data for yesterday - hold position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data yet - hold position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0