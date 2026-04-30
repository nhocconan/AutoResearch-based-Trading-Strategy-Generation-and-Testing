#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Uses 1w EMA34 for stable weekly trend direction and requires volume > 2.0x 20-period average.
# Trades only during 08-20 UTC session to avoid low-liquidity periods.
# Designed for low trade frequency (~10-20 trades/year) to minimize fee drag.
# Camarilla levels provide weekly support/resistance that work in both trending and ranging markets.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        
        # Volume confirmation: volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_confirm = False
        
        # Calculate Camarilla levels for current week using previous week's OHLC
        # Need to group by week to get previous week's OHLC
        if i >= 7:  # Need at least 7 days of data for previous week
            # Get timestamp of current bar
            curr_time = prices.iloc[i]["open_time"]
            # Get start of current week (Monday 00:00 UTC)
            curr_week_start = curr_time - pd.Timedelta(days=curr_time.weekday())
            curr_week_start = curr_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            # Get start of previous week
            prev_week_start = curr_week_start - pd.Timedelta(weeks=1)
            # Get end of previous week (Sunday 23:59:59.999 UTC)
            prev_week_end = curr_week_start - pd.Timedelta(microseconds=1)
            
            # Filter prices for previous week
            mask = (prices["open_time"] >= prev_week_start) & (prices["open_time"] <= prev_week_end)
            if mask.any():
                prev_week_data = prices.loc[mask]
                if len(prev_week_data) > 0:
                    prev_high = prev_week_data["high"].max()
                    prev_low = prev_week_data["low"].min()
                    prev_close = prev_week_data["close"].iloc[-1]
                    
                    # Calculate Camarilla levels
                    range_val = prev_high - prev_low
                    if range_val > 0:
                        camarilla_r3 = prev_close + (range_val * 1.1 / 4)
                        camarilla_s3 = prev_close - (range_val * 1.1 / 4)
                    else:
                        camarilla_r3 = curr_close
                        camarilla_s3 = curr_close
                else:
                    camarilla_r3 = curr_close
                    camarilla_s3 = curr_close
            else:
                camarilla_r3 = curr_close
                camarilla_s3 = curr_close
        else:
            camarilla_r3 = curr_close
            camarilla_s3 = curr_close
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3, 1w EMA34 uptrend, volume spike
            if (curr_close > camarilla_r3 and 
                curr_close > curr_ema_34_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Camarilla S3, 1w EMA34 downtrend, volume spike
            elif (curr_close < camarilla_s3 and 
                  curr_close < curr_ema_34_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below Camarilla S3
            if curr_close < camarilla_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Camarilla R3
            if curr_close > camarilla_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals