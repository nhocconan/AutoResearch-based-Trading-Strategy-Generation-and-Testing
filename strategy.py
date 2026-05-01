#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Uses 1d EMA50 for trend alignment (HTF direction) and 1h volume spike for entry timing.
# Long when price breaks above R3 and above 1d EMA50 with volume > 1.5x 20-period average.
# Short when price breaks below S3 and below 1d EMA50 with volume > 1.5x 20-period average.
# Exit on opposite Camarilla level (R1/S1) break for tighter risk control.
# Session filter (08-20 UTC) reduces noise. Discrete sizing 0.20 minimizes fee churn.
# Target: 15-30 trades/year by using 1d for signal direction and 1h only for entry timing.

name = "1h_Camarilla_R3S3_Breakout_1dEMA50_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d data
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h volume average (20-period) for volume spike filter
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA and volume average
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_50_aligned[i]
        curr_volume = volume[i]
        curr_volume_avg = volume_avg[i]
        
        # Skip if volume data not ready
        if np.isnan(curr_volume_avg):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = curr_volume > (1.5 * curr_volume_avg)
        
        # Calculate Camarilla levels for current day using previous day's OHLC
        if i >= 24:  # Need at least 24 bars (1 day) of 1h data for previous day
            # Get timestamp of current bar
            curr_time = prices.iloc[i]["open_time"]
            # Get start of current day (00:00 UTC)
            curr_day_start = curr_time.replace(hour=0, minute=0, second=0, microsecond=0)
            # Get start of previous day
            prev_day_start = curr_day_start - pd.Timedelta(days=1)
            # Get end of previous day (23:59:59.999 UTC)
            prev_day_end = curr_day_start - pd.Timedelta(microseconds=1)
            
            # Filter prices for previous day
            mask = (prices["open_time"] >= prev_day_start) & (prices["open_time"] <= prev_day_end)
            if mask.any():
                prev_day_data = prices.loc[mask]
                if len(prev_day_data) > 0:
                    prev_high = prev_day_data["high"].max()
                    prev_low = prev_day_data["low"].min()
                    prev_close = prev_day_data["close"].iloc[-1]
                    
                    # Calculate Camarilla levels
                    range_val = prev_high - prev_low
                    if range_val > 0:
                        camarilla_r3 = prev_close + (range_val * 1.1 / 4)   # R3 level
                        camarilla_s3 = prev_close - (range_val * 1.1 / 4)   # S3 level
                        camarilla_r1 = prev_close + (range_val * 1.1 / 12)  # R1 level (exit)
                        camarilla_s1 = prev_close - (range_val * 1.1 / 12)  # S1 level (exit)
                    else:
                        camarilla_r3 = curr_close
                        camarilla_s3 = curr_close
                        camarilla_r1 = curr_close
                        camarilla_s1 = curr_close
                else:
                    camarilla_r3 = curr_close
                    camarilla_s3 = curr_close
                    camarilla_r1 = curr_close
                    camarilla_s1 = curr_close
            else:
                camarilla_r3 = curr_close
                camarilla_s3 = curr_close
                camarilla_r1 = curr_close
                camarilla_s1 = curr_close
        else:
            camarilla_r3 = curr_close
            camarilla_s3 = curr_close
            camarilla_r1 = curr_close
            camarilla_s1 = curr_close
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, price above 1d EMA50, volume confirmation
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema and 
                volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3, price below 1d EMA50, volume confirmation
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema and 
                  volume_confirm):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S1 (tighter stop)
            if curr_close < camarilla_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R1 (tighter stop)
            if curr_close > camarilla_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals