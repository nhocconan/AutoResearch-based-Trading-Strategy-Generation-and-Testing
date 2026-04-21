#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: On 4h timeframe, price breaking above Camarilla R1 or below S1 indicates short-term breakout, with 12h EMA50 trend filter and volume spike confirmation. Uses discrete sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMA50 for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for today (using previous day's OHLC)
        # Need to group by date to get daily OHLC
        if i >= 24:  # Need at least 24 hours of 4h data (6 bars) to form a day
            # Get indices for today's 4h bars (starting from 00:00 UTC)
            today_start = prices.iloc[i]['open_time'].normalize()
            today_bars = prices[prices['open_time'] >= today_start]
            if len(today_bars) >= 6:  # At least 6 bars (24h) have passed today
                # Use yesterday's OHLC (excluding today)
                yesterday = prices[prices['open_time'] < today_start]
                if len(yesterday) >= 6:  # Need full yesterday
                    y_high = yesterday['high'].max()
                    y_low = yesterday['low'].min()
                    y_close = yesterday['close'].iloc[-1]
                    
                    # Camarilla levels
                    R1 = y_close + (y_high - y_low) * 1.1 / 12
                    S1 = y_close - (y_high - y_low) * 1.1 / 12
                    
                    price_close = prices['close'].iloc[i]
                    price_high = prices['high'].iloc[i]
                    price_low = prices['low'].iloc[i]
                    volume = prices['volume'].iloc[i]
                    ema_50 = ema_50_12h_aligned[i]
                    
                    # Volume spike: current volume > 1.5 * average volume of last 24 periods (6h)
                    if i >= 30:  # Need enough history for volume average
                        vol_avg = prices['volume'].iloc[i-24:i].mean()
                        vol_spike = volume > 1.5 * vol_avg if not np.isnan(vol_avg) else False
                    else:
                        vol_spike = False
                    
                    if position == 0:
                        # Long: price breaks above R1 + above 12h EMA50 + volume spike
                        if price_high > R1 and price_close > ema_50 and vol_spike:
                            signals[i] = 0.25
                            position = 1
                        # Short: price breaks below S1 + below 12h EMA50 + volume spike
                        elif price_low < S1 and price_close < ema_50 and vol_spike:
                            signals[i] = -0.25
                            position = -1
                    
                    elif position != 0:
                        # Exit when price re-enters Camarilla levels or trend weakens
                        if position == 1:
                            if price_low < R1 or price_close < ema_50:
                                signals[i] = 0.0
                                position = 0
                            else:
                                signals[i] = 0.25
                        else:  # position == -1
                            if price_high > S1 or price_close > ema_50:
                                signals[i] = 0.0
                                position = 0
                            else:
                                signals[i] = -0.25
                    continue  # Skip to next iteration after processing
        
        # Default: hold current position or flat
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0