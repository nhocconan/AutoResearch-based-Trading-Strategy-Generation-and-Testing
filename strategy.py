#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike
# Camarilla levels provide precise intraday support/resistance from prior session.
# Breakout above R3 or below S3 with 12h trend alignment captures momentum with low false breaks.
# Volume spike confirms institutional participation. Designed for 20-40 trades/year on 4h to minimize fee drag.
# Works in bull markets via buying R3 breakouts in uptrends and bear markets via selling S3 breakdowns in downtrends.

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Need at least 3 bars for Camarilla calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using prior day's OHLC
        # Camarilla requires prior day's data, so we need to get prior day's high/low/close
        # Since we're on 4h timeframe, we need to aggregate to daily
        # But per rules, we must use mtf_data helper - so we'll get 1d data separately
        if i >= 6:  # Need at least 6*4h = 24h to form a full day
            # Get 1d data for Camarilla calculation (using mtf_data helper)
            df_1d = get_htf_data(prices, '1d')
            if len(df_1d) < 1:
                # Not enough daily data yet
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
            
            # We need prior day's OHLC (not current forming day)
            # So we use the second-to-last completed daily bar
            if len(df_1d) >= 2:
                prior_high = df_1d['high'].iloc[-2]
                prior_low = df_1d['low'].iloc[-2]
                prior_close = df_1d['close'].iloc[-2]
            else:
                # Not enough prior days
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
            
            # Calculate Camarilla levels
            range_val = prior_high - prior_low
            if range_val <= 0:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
            
            # Camarilla R3 and S3 levels
            r3 = prior_close + (range_val * 1.1 / 4)
            s3 = prior_close - (range_val * 1.1 / 4)
            
            # Volume confirmation: 20-period EMA on 4h
            if i >= 19:
                vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
            else:
                vol_ema_20 = volume[i]
            volume_spike = volume[i] > (1.5 * vol_ema_20)
            
            # Camarilla breakout conditions
            breakout_up = close[i] > r3
            breakout_down = close[i] < s3
            
            if position == 0:
                # Long: bullish breakout above R3 in 12h uptrend with volume spike
                if breakout_up and ema_34_12h_aligned[i] < close[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish breakdown below S3 in 12h downtrend with volume spike
                elif breakout_down and ema_34_12h_aligned[i] > close[i] and volume_spike:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: price returns to R3 level or loses 12h uptrend
                if close[i] < r3 or ema_34_12h_aligned[i] >= close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to S3 level or loses 12h downtrend
                if close[i] > s3 or ema_34_12h_aligned[i] <= close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Not enough data for Camarilla calculation yet
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals